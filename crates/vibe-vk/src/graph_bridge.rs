//! Bridges the render graph's synthesized barriers to the Vulkan encoder.
//!
//! The graph reasons in terms of [`ResourceUsage`] and pass ids; the encoder
//! needs Vulkan stages, access flags and image handles. This module is the
//! translation, and it is where a resource's role in the frame decides its
//! barrier: a colour target written by the render pass and read by a post pass
//! needs a different transition than a storage buffer a compute pass writes.

use ash::vk;

use vibe_graph::{Barrier as GraphBarrier, CompiledGraph, ResourceResolver};

use crate::barrier::{Access, Barrier, BarrierEncoder, ResourceKind, Stage};

/// How a resource is used by the frame, which decides its barrier shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ResourceRole {
    /// A render target written by a raster pass.
    ColorTarget,
    /// A depth or stencil target.
    DepthTarget,
    /// A buffer or image read by a compute or fragment shader.
    ShaderResource,
    /// A buffer written by a transfer.
    TransferDst,
    /// A buffer read by a transfer.
    TransferSrc,
    /// Host-visible memory the CPU writes each frame.
    HostVisible,
}

impl ResourceRole {
    /// The stage that writes this kind of resource.
    pub fn write_stage(self) -> Stage {
        match self {
            ResourceRole::ColorTarget => Stage::ColorAttachmentOutput,
            ResourceRole::DepthTarget => Stage::LateFragmentTests,
            ResourceRole::ShaderResource => Stage::ComputeShader,
            ResourceRole::TransferDst | ResourceRole::TransferSrc => Stage::Transfer,
            ResourceRole::HostVisible => Stage::None,
        }
    }

    /// The access a write to this kind of resource performs.
    pub fn write_access(self) -> Access {
        match self {
            ResourceRole::ColorTarget => Access::ColorAttachmentWrite,
            ResourceRole::DepthTarget => Access::DepthStencilAttachmentWrite,
            ResourceRole::ShaderResource => Access::ShaderWrite,
            ResourceRole::TransferDst | ResourceRole::TransferSrc => Access::TransferWrite,
            ResourceRole::HostVisible => Access::HostWrite,
        }
    }

    /// The stage that reads this kind of resource.
    pub fn read_stage(self) -> Stage {
        match self {
            ResourceRole::ColorTarget | ResourceRole::DepthTarget => Stage::FragmentShader,
            ResourceRole::ShaderResource => Stage::FragmentShader,
            ResourceRole::TransferDst | ResourceRole::TransferSrc => Stage::Transfer,
            ResourceRole::HostVisible => Stage::Transfer,
        }
    }

    /// The access a read of this kind of resource performs.
    pub fn read_access(self) -> Access {
        match self {
            ResourceRole::ColorTarget => Access::ShaderRead,
            ResourceRole::DepthTarget => Access::ShaderRead,
            ResourceRole::ShaderResource => Access::ShaderRead,
            ResourceRole::TransferDst | ResourceRole::TransferSrc => Access::TransferRead,
            ResourceRole::HostVisible => Access::HostRead,
        }
    }

    /// True when this role transitions an image layout.
    pub fn is_image(self) -> bool {
        matches!(
            self,
            ResourceRole::ColorTarget | ResourceRole::DepthTarget | ResourceRole::ShaderResource
        )
    }

    /// The image aspect for this role.
    pub fn aspect(self) -> vk::ImageAspectFlags {
        match self {
            ResourceRole::DepthTarget => vk::ImageAspectFlags::DEPTH,
            _ => vk::ImageAspectFlags::COLOR,
        }
    }
}

/// The live Vulkan object behind a graph resource.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BackendObject {
    /// An image, with the role the frame gives it.
    Image {
        handle: vk::Image,
        role: ResourceRole,
    },
    /// A buffer, with the role the frame gives it.
    Buffer {
        handle: vk::Buffer,
        role: ResourceRole,
    },
}

/// Resolves graph handles to Vulkan objects and knows each one's role.
pub trait ObjectResolver: ResourceResolver {
    /// The Vulkan object behind a handle, if it is live.
    fn object(&self, resource: vibe_rhi::ResourceHandle) -> Option<BackendObject>;
}

/// Fill an encoder with every barrier a compiled graph needs.
///
/// Only image and buffer resources produce barriers; a resource whose object
/// cannot be resolved is skipped rather than emitting a barrier against a null
/// handle, which the driver would reject.
///
/// # Errors
///
/// Returns an error only if the caller asked for strict mode and a handle
/// failed to resolve, which means the graph is being run against a world it
/// does not describe.
pub fn encode_graph<R: ObjectResolver>(
    graph: &CompiledGraph,
    resolver: &R,
    strict: bool,
) -> Result<usize, crate::context::VkError> {
    let mut encoder = BarrierEncoder::new();
    let mut unresolved = 0usize;

    for pass in &graph.passes {
        for gb in &pass.barriers {
            let Some(object) = resolver.object(gb.resource) else {
                unresolved += 1;
                continue;
            };
            let barrier = translate(gb, object);
            encoder.push(barrier);
        }
    }

    if strict && unresolved > 0 {
        return Err(crate::context::VkError::Swapchain(format!(
            "{unresolved} graph resource(s) could not be resolved to a vulkan object"
        )));
    }
    Ok(encoder.len())
}

/// Fill an encoder with every barrier a compiled graph needs, on a tier.
pub fn encode_graph_for_tier<R: ObjectResolver>(
    graph: &CompiledGraph,
    resolver: &R,
    tier: crate::sync::SyncTier,
) -> BarrierEncoder {
    let mut encoder = BarrierEncoder::for_tier(tier);
    for pass in &graph.passes {
        for gb in &pass.barriers {
            let Some(object) = resolver.object(gb.resource) else {
                continue;
            };
            encoder.push(translate(gb, object));
        }
    }
    encoder
}

/// Turn one graph barrier into a Vulkan barrier for a known object.
pub fn translate(graph_barrier: &GraphBarrier, object: BackendObject) -> Barrier {
    let (role, is_image) = match object {
        BackendObject::Image { role, .. } => (role, true),
        BackendObject::Buffer { role, .. } => (role, false),
    };

    // The graph's Read/Write/Discard become the role's read and write sides.
    let (src_stage, src_access) = match graph_barrier.from {
        vibe_rhi::ResourceUsage::Read => (role.read_stage(), role.read_access()),
        vibe_rhi::ResourceUsage::Write | vibe_rhi::ResourceUsage::ReadWrite => {
            (role.write_stage(), role.write_access())
        }
        vibe_rhi::ResourceUsage::Discard => (Stage::None, Access::None),
    };
    let (dst_stage, dst_access) = match graph_barrier.to {
        vibe_rhi::ResourceUsage::Read => (role.read_stage(), role.read_access()),
        vibe_rhi::ResourceUsage::Write | vibe_rhi::ResourceUsage::ReadWrite => {
            (role.write_stage(), role.write_access())
        }
        vibe_rhi::ResourceUsage::Discard => (Stage::None, Access::None),
    };

    let kind = if is_image {
        ResourceKind::Image
    } else {
        ResourceKind::Buffer
    };
    let mut barrier = match object {
        BackendObject::Image { handle, .. } => {
            Barrier::image(handle, role.aspect()).with_kind(kind)
        }
        BackendObject::Buffer { handle, .. } => Barrier::buffer(handle).with_kind(kind),
    };
    barrier.src_stage = src_stage;
    barrier.src_access = src_access;
    barrier.dst_stage = dst_stage;
    barrier.dst_access = dst_access;
    barrier
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sync::SyncTier;
    use ash::vk::Handle as _;
    use vibe_graph::{PassGraph, ResourceRegistry};
    use vibe_rhi::ResourceHandle;

    fn h(i: u32) -> ResourceHandle {
        ResourceHandle {
            index: i,
            generation: 0,
        }
    }

    fn dummy_image() -> vk::Image {
        vk::Image::from_raw(0x1000)
    }

    fn dummy_buffer() -> vk::Buffer {
        vk::Buffer::from_raw(0x2000)
    }

    struct FakeResolver {
        objects: Vec<(ResourceHandle, BackendObject)>,
    }

    impl FakeResolver {
        fn new(objects: Vec<(ResourceHandle, BackendObject)>) -> FakeResolver {
            FakeResolver { objects }
        }
    }

    impl vibe_graph::ResourceResolver for FakeResolver {
        fn resolve(&self, resource: ResourceHandle) -> Option<u64> {
            self.objects
                .iter()
                .find(|(r, _)| *r == resource)
                .map(|(r, _)| r.index() as u64)
        }

        fn begin_pass(
            &mut self,
            _pass: vibe_graph::PassId,
            _barriers: &[vibe_graph::Barrier],
        ) -> Result<(), vibe_graph::GraphError> {
            Ok(())
        }

        fn end_pass(&mut self, _pass: vibe_graph::PassId) -> Result<(), vibe_graph::GraphError> {
            Ok(())
        }
    }

    impl ObjectResolver for FakeResolver {
        fn object(&self, resource: ResourceHandle) -> Option<BackendObject> {
            self.objects
                .iter()
                .find(|(r, _)| *r == resource)
                .map(|(_, o)| *o)
        }
    }

    fn color_resolver() -> FakeResolver {
        FakeResolver::new(vec![(
            h(0),
            BackendObject::Image {
                handle: dummy_image(),
                role: ResourceRole::ColorTarget,
            },
        )])
    }

    fn write_read_graph() -> CompiledGraph {
        let mut r = ResourceRegistry::new();
        let target = r.create(
            vibe_graph::ResourceDesc::Texture {
                width: 4,
                height: 4,
                format: 0,
            },
            "color",
        );
        let mut g = PassGraph::new();
        g.add_pass("draw").write(target).finish();
        g.add_pass("post").read(target).finish();
        g.compile().unwrap()
    }

    #[test]
    fn color_target_writes_from_the_color_attachment_stage() {
        assert_eq!(
            ResourceRole::ColorTarget.write_stage(),
            Stage::ColorAttachmentOutput
        );
        assert_eq!(
            ResourceRole::ColorTarget.write_access(),
            Access::ColorAttachmentWrite
        );
    }

    #[test]
    fn depth_target_writes_from_late_fragment_tests() {
        assert_eq!(
            ResourceRole::DepthTarget.write_stage(),
            Stage::LateFragmentTests
        );
        assert_eq!(
            ResourceRole::DepthTarget.write_access(),
            Access::DepthStencilAttachmentWrite
        );
    }

    #[test]
    fn targets_are_read_by_the_fragment_shader() {
        for role in [ResourceRole::ColorTarget, ResourceRole::DepthTarget] {
            assert_eq!(role.read_stage(), Stage::FragmentShader, "{role:?}");
            assert_eq!(role.read_access(), Access::ShaderRead, "{role:?}");
        }
    }

    #[test]
    fn transfer_roles_use_the_transfer_stage() {
        assert_eq!(ResourceRole::TransferDst.write_stage(), Stage::Transfer);
        assert_eq!(ResourceRole::TransferSrc.read_stage(), Stage::Transfer);
    }

    #[test]
    fn host_visible_writes_have_no_gpu_stage() {
        assert_eq!(ResourceRole::HostVisible.write_stage(), Stage::None);
        assert_eq!(ResourceRole::HostVisible.write_access(), Access::HostWrite);
    }

    #[test]
    fn only_targets_and_shader_resources_are_images() {
        assert!(ResourceRole::ColorTarget.is_image());
        assert!(ResourceRole::DepthTarget.is_image());
        assert!(ResourceRole::ShaderResource.is_image());
        assert!(!ResourceRole::TransferDst.is_image());
        assert!(!ResourceRole::HostVisible.is_image());
    }

    #[test]
    fn depth_uses_the_depth_aspect() {
        assert_eq!(
            ResourceRole::DepthTarget.aspect(),
            vk::ImageAspectFlags::DEPTH
        );
        assert_eq!(
            ResourceRole::ColorTarget.aspect(),
            vk::ImageAspectFlags::COLOR
        );
    }

    #[test]
    fn translating_a_write_produces_the_writes_stage_and_access() {
        let gb = vibe_graph::Barrier {
            resource: h(0),
            pass: vibe_graph::PassId(0),
            from: vibe_rhi::ResourceUsage::Discard,
            to: vibe_rhi::ResourceUsage::Write,
        };
        let b = translate(
            &gb,
            BackendObject::Image {
                handle: dummy_image(),
                role: ResourceRole::ColorTarget,
            },
        );
        assert_eq!(b.dst_stage, Stage::ColorAttachmentOutput);
        assert_eq!(b.dst_access, Access::ColorAttachmentWrite);
        assert_eq!(b.src_stage, Stage::None, "a discard has no source");
    }

    #[test]
    fn translating_a_read_produces_the_reads_stage_and_access() {
        let gb = vibe_graph::Barrier {
            resource: h(0),
            pass: vibe_graph::PassId(1),
            from: vibe_rhi::ResourceUsage::Write,
            to: vibe_rhi::ResourceUsage::Read,
        };
        let b = translate(
            &gb,
            BackendObject::Image {
                handle: dummy_image(),
                role: ResourceRole::ColorTarget,
            },
        );
        assert_eq!(b.src_stage, Stage::ColorAttachmentOutput);
        assert_eq!(b.src_access, Access::ColorAttachmentWrite);
        assert_eq!(b.dst_stage, Stage::FragmentShader);
        assert_eq!(b.dst_access, Access::ShaderRead);
    }

    #[test]
    fn translating_produces_the_right_kind() {
        let gb = vibe_graph::Barrier {
            resource: h(0),
            pass: vibe_graph::PassId(0),
            from: vibe_rhi::ResourceUsage::Discard,
            to: vibe_rhi::ResourceUsage::Write,
        };
        let image = translate(
            &gb,
            BackendObject::Image {
                handle: dummy_image(),
                role: ResourceRole::ColorTarget,
            },
        );
        assert_eq!(image.kind, ResourceKind::Image);
        assert_eq!(image.handle, dummy_image());

        let buffer = translate(
            &gb,
            BackendObject::Buffer {
                handle: dummy_buffer(),
                role: ResourceRole::ShaderResource,
            },
        );
        assert_eq!(buffer.kind, ResourceKind::Buffer);
        assert_eq!(buffer.buffer, dummy_buffer());
    }

    #[test]
    fn a_depth_barrier_uses_the_depth_aspect() {
        let gb = vibe_graph::Barrier {
            resource: h(0),
            pass: vibe_graph::PassId(0),
            from: vibe_rhi::ResourceUsage::Discard,
            to: vibe_rhi::ResourceUsage::Write,
        };
        let b = translate(
            &gb,
            BackendObject::Image {
                handle: dummy_image(),
                role: ResourceRole::DepthTarget,
            },
        );
        assert_eq!(b.aspect, vk::ImageAspectFlags::DEPTH);
    }

    #[test]
    fn encoding_a_graph_produces_one_barrier_per_transition() {
        let graph = write_read_graph();
        let count = encode_graph(&graph, &color_resolver(), true).unwrap();
        assert_eq!(count, 1, "a write then a read is one transition");
    }

    #[test]
    fn encoding_bind_the_encoder_to_the_tier() {
        let graph = write_read_graph();
        for tier in [SyncTier::Legacy, SyncTier::Sync2, SyncTier::Sync2Timeline] {
            let enc = encode_graph_for_tier(&graph, &color_resolver(), tier);
            assert_eq!(enc.len(), 1, "tier {tier} must produce the same barrier");
        }
    }

    #[test]
    fn encoded_barriers_carry_the_right_stages() {
        let graph = write_read_graph();
        let enc = encode_graph_for_tier(&graph, &color_resolver(), SyncTier::Sync2);
        let b = enc.barriers()[0];
        assert_eq!(b.src_stage, Stage::ColorAttachmentOutput);
        assert_eq!(b.dst_stage, Stage::FragmentShader);
    }

    #[test]
    fn unresolvable_resources_are_skipped_in_lenient_mode() {
        let graph = write_read_graph();
        let empty = FakeResolver::new(vec![]);
        let count = encode_graph(&graph, &empty, false).unwrap();
        assert_eq!(
            count, 0,
            "an unresolvable handle must not emit a null barrier"
        );
    }

    #[test]
    fn unresolvable_resources_are_an_error_in_strict_mode() {
        let graph = write_read_graph();
        let empty = FakeResolver::new(vec![]);
        let err = encode_graph(&graph, &empty, true).unwrap_err();
        assert!(matches!(err, crate::context::VkError::Swapchain(_)));
        assert!(err.to_string().contains("could not be resolved"));
    }

    #[test]
    fn an_empty_graph_encodes_to_nothing() {
        let graph = PassGraph::new().compile().unwrap();
        let enc = encode_graph_for_tier(&graph, &color_resolver(), SyncTier::Sync2);
        assert!(enc.is_empty());
    }

    #[test]
    fn a_graph_with_no_transitions_encodes_to_nothing() {
        let mut r = ResourceRegistry::new();
        let target = r.create(
            vibe_graph::ResourceDesc::Texture {
                width: 1,
                height: 1,
                format: 0,
            },
            "t",
        );
        let mut g = PassGraph::new();
        g.add_pass("only").discard(target).finish();
        let resolver = FakeResolver::new(vec![(
            h(0),
            BackendObject::Image {
                handle: dummy_image(),
                role: ResourceRole::ColorTarget,
            },
        )]);
        let enc = encode_graph_for_tier(&g.compile().unwrap(), &resolver, SyncTier::Sync2);
        assert!(enc.is_empty(), "a discard-to-nothing needs no barrier");
    }
}
