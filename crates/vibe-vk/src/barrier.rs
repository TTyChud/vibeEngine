//! One barrier encoder for all three sync tiers.
//!
//! The engine has a single call site for barriers, and it emits
//! `vkCmdPipelineBarrier2` on the Sync2 tiers and `vkCmdPipelineBarrier` on
//! Legacy. Keeping the two spellings behind one type is the point: a
//! `BarrierEncoder` built for a Legacy device and one built for a Sync2 device
//! present the same API to the renderer, and the graph's synthesized barriers
//! are written once.
//!
//! The tier decision and the barrier *contents* are pure logic and are tested
//! without a device. Recording itself needs a live `ash::Device`.

use ash::vk;

use crate::sync::SyncTier;

/// Which pipeline stage a barrier waits on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Stage {
    /// No work; used for a barrier that only needs a memory dependency.
    None,
    TopOfPipe,
    VertexInput,
    VertexShader,
    FragmentShader,
    EarlyFragmentTests,
    LateFragmentTests,
    ColorAttachmentOutput,
    ComputeShader,
    Transfer,
    BottomOfPipe,
    /// All commands in the previous submission.
    AllCommands,
}

impl Stage {
    /// The `VK_PIPELINE_STAGE_*` bit(s) this stage maps to.
    pub const fn flags(self) -> vk::PipelineStageFlags2 {
        match self {
            Stage::None => vk::PipelineStageFlags2::empty(),
            Stage::TopOfPipe => vk::PipelineStageFlags2::TOP_OF_PIPE,
            Stage::VertexInput => vk::PipelineStageFlags2::VERTEX_INPUT,
            Stage::VertexShader => vk::PipelineStageFlags2::VERTEX_SHADER,
            Stage::FragmentShader => vk::PipelineStageFlags2::FRAGMENT_SHADER,
            Stage::EarlyFragmentTests => vk::PipelineStageFlags2::EARLY_FRAGMENT_TESTS,
            Stage::LateFragmentTests => vk::PipelineStageFlags2::LATE_FRAGMENT_TESTS,
            Stage::ColorAttachmentOutput => vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT,
            Stage::ComputeShader => vk::PipelineStageFlags2::COMPUTE_SHADER,
            Stage::Transfer => vk::PipelineStageFlags2::TRANSFER,
            Stage::BottomOfPipe => vk::PipelineStageFlags2::BOTTOM_OF_PIPE,
            Stage::AllCommands => vk::PipelineStageFlags2::ALL_COMMANDS,
        }
    }

    /// The 1.0-stage equivalent, for the Legacy path.
    ///
    /// `COLOR_ATTACHMENT_OUTPUT` and the fragment stages have no 1.0
    /// equivalent, so they map to the top of the pipe, which is correct there:
    /// 1.0 barriers are coarse and the driver places them conservatively.
    pub const fn legacy_flags(self) -> vk::PipelineStageFlags {
        match self {
            Stage::None => vk::PipelineStageFlags::empty(),
            Stage::TopOfPipe => vk::PipelineStageFlags::TOP_OF_PIPE,
            Stage::VertexInput => vk::PipelineStageFlags::VERTEX_INPUT,
            Stage::VertexShader => vk::PipelineStageFlags::VERTEX_SHADER,
            // 1.0 has no fragment stages; TOP_OF_PIPE is the safe equivalent.
            Stage::FragmentShader
            | Stage::EarlyFragmentTests
            | Stage::LateFragmentTests
            | Stage::ColorAttachmentOutput => vk::PipelineStageFlags::TOP_OF_PIPE,
            Stage::ComputeShader => vk::PipelineStageFlags::COMPUTE_SHADER,
            Stage::Transfer => vk::PipelineStageFlags::TRANSFER,
            Stage::BottomOfPipe => vk::PipelineStageFlags::BOTTOM_OF_PIPE,
            Stage::AllCommands => vk::PipelineStageFlags::ALL_COMMANDS,
        }
    }
}

/// A memory access type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Access {
    /// No access; the barrier exists only for an execution dependency.
    None,
    IndirectCommandRead,
    IndexRead,
    VertexAttributeRead,
    UniformRead,
    ShaderRead,
    ShaderWrite,
    ColorAttachmentRead,
    ColorAttachmentWrite,
    DepthStencilAttachmentRead,
    DepthStencilAttachmentWrite,
    TransferRead,
    TransferWrite,
    HostRead,
    HostWrite,
    MemoryRead,
    MemoryWrite,
}

impl Access {
    /// The `VK_ACCESS_*` bit(s).
    pub const fn flags2(self) -> vk::AccessFlags2 {
        match self {
            Access::None => vk::AccessFlags2::empty(),
            Access::IndirectCommandRead => vk::AccessFlags2::INDIRECT_COMMAND_READ,
            Access::IndexRead => vk::AccessFlags2::INDEX_READ,
            Access::VertexAttributeRead => vk::AccessFlags2::VERTEX_ATTRIBUTE_READ,
            Access::UniformRead => vk::AccessFlags2::UNIFORM_READ,
            Access::ShaderRead => vk::AccessFlags2::SHADER_READ,
            Access::ShaderWrite => vk::AccessFlags2::SHADER_WRITE,
            Access::ColorAttachmentRead => vk::AccessFlags2::COLOR_ATTACHMENT_READ,
            Access::ColorAttachmentWrite => vk::AccessFlags2::COLOR_ATTACHMENT_WRITE,
            Access::DepthStencilAttachmentRead => vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_READ,
            Access::DepthStencilAttachmentWrite => vk::AccessFlags2::DEPTH_STENCIL_ATTACHMENT_WRITE,
            Access::TransferRead => vk::AccessFlags2::TRANSFER_READ,
            Access::TransferWrite => vk::AccessFlags2::TRANSFER_WRITE,
            Access::HostRead => vk::AccessFlags2::HOST_READ,
            Access::HostWrite => vk::AccessFlags2::HOST_WRITE,
            Access::MemoryRead => vk::AccessFlags2::MEMORY_READ,
            Access::MemoryWrite => vk::AccessFlags2::MEMORY_WRITE,
        }
    }

    /// The 1.0 access bits.
    pub const fn legacy_flags(self) -> vk::AccessFlags {
        match self {
            Access::None => vk::AccessFlags::empty(),
            Access::IndirectCommandRead => vk::AccessFlags::INDIRECT_COMMAND_READ,
            Access::IndexRead => vk::AccessFlags::INDEX_READ,
            Access::VertexAttributeRead => vk::AccessFlags::VERTEX_ATTRIBUTE_READ,
            Access::UniformRead => vk::AccessFlags::UNIFORM_READ,
            Access::ShaderRead => vk::AccessFlags::SHADER_READ,
            Access::ShaderWrite => vk::AccessFlags::SHADER_WRITE,
            Access::ColorAttachmentRead => vk::AccessFlags::COLOR_ATTACHMENT_READ,
            Access::ColorAttachmentWrite => vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
            Access::DepthStencilAttachmentRead => vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ,
            Access::DepthStencilAttachmentWrite => vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
            Access::TransferRead => vk::AccessFlags::TRANSFER_READ,
            Access::TransferWrite => vk::AccessFlags::TRANSFER_WRITE,
            Access::HostRead => vk::AccessFlags::HOST_READ,
            Access::HostWrite => vk::AccessFlags::HOST_WRITE,
            Access::MemoryRead => vk::AccessFlags::MEMORY_READ,
            Access::MemoryWrite => vk::AccessFlags::MEMORY_WRITE,
        }
    }
}

/// The kind of resource a barrier applies to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ResourceKind {
    /// A whole-device memory barrier.
    Memory,
    /// A range of a buffer.
    Buffer,
    /// A subresource range of an image.
    Image,
}

/// One memory barrier, in backend-neutral terms.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Barrier {
    /// What kind of resource this covers.
    pub kind: ResourceKind,
    /// The image or buffer being transitioned, if not a whole-device barrier.
    pub handle: vk::Image,
    /// The buffer being transitioned, if this is a buffer barrier.
    pub buffer: vk::Buffer,
    /// Byte or texel offset into the resource.
    pub offset: u64,
    /// Size of the range; `u64::MAX` means "to the end".
    pub size: u64,
    /// Source mip level, for images.
    pub base_mip: u32,
    /// Number of mip levels, for images.
    pub mip_count: u32,
    /// Source array layer, for images.
    pub base_layer: u32,
    /// Number of array layers, for images.
    pub layer_count: u32,
    /// Aspect to transition, for depth/stencil images.
    pub aspect: vk::ImageAspectFlags,
    /// Stage the producing work runs in.
    pub src_stage: Stage,
    /// Access the producing work performs.
    pub src_access: Access,
    /// Stage the consuming work runs in.
    pub dst_stage: Stage,
    /// Access the consuming work performs.
    pub dst_access: Access,
}

impl Barrier {
    /// A whole-device memory barrier.
    pub fn memory(
        src_stage: Stage,
        src_access: Access,
        dst_stage: Stage,
        dst_access: Access,
    ) -> Barrier {
        Barrier {
            kind: ResourceKind::Memory,
            handle: vk::Image::null(),
            buffer: vk::Buffer::null(),
            offset: 0,
            size: vk::WHOLE_SIZE,
            base_mip: 0,
            mip_count: vk::REMAINING_MIP_LEVELS,
            base_layer: 0,
            layer_count: vk::REMAINING_ARRAY_LAYERS,
            aspect: vk::ImageAspectFlags::empty(),
            src_stage,
            src_access,
            dst_stage,
            dst_access,
        }
    }

    /// A buffer range barrier covering the whole buffer.
    pub fn buffer(buffer: vk::Buffer) -> Barrier {
        Barrier {
            kind: ResourceKind::Buffer,
            handle: vk::Image::null(),
            buffer,
            offset: 0,
            size: vk::WHOLE_SIZE,
            base_mip: 0,
            mip_count: vk::REMAINING_MIP_LEVELS,
            base_layer: 0,
            layer_count: vk::REMAINING_ARRAY_LAYERS,
            aspect: vk::ImageAspectFlags::empty(),
            src_stage: Stage::None,
            src_access: Access::None,
            dst_stage: Stage::None,
            dst_access: Access::None,
        }
    }

    /// An image subresource barrier for one mip of one layer.
    pub fn image(image: vk::Image, aspect: vk::ImageAspectFlags) -> Barrier {
        Barrier {
            kind: ResourceKind::Image,
            handle: image,
            buffer: vk::Buffer::null(),
            offset: 0,
            size: vk::WHOLE_SIZE,
            base_mip: 0,
            mip_count: 1,
            base_layer: 0,
            layer_count: 1,
            aspect,
            src_stage: Stage::None,
            src_access: Access::None,
            dst_stage: Stage::None,
            dst_access: Access::None,
        }
    }

    /// Override the resource kind, which the constructors infer.
    pub fn with_kind(mut self, kind: ResourceKind) -> Barrier {
        self.kind = kind;
        self
    }

    /// Set the source stage and access.
    pub fn src(mut self, stage: Stage, access: Access) -> Barrier {
        self.src_stage = stage;
        self.src_access = access;
        self
    }

    /// Set the destination stage and access.
    pub fn dst(mut self, stage: Stage, access: Access) -> Barrier {
        self.dst_stage = stage;
        self.dst_access = access;
        self
    }

    /// True when the barrier changes nothing, so it can be dropped.
    pub fn is_noop(&self) -> bool {
        self.src_stage == Stage::None
            && self.src_access == Access::None
            && self.dst_stage == Stage::None
            && self.dst_access == Access::None
    }

    fn to_memory_barrier(&self) -> vk::MemoryBarrier<'_> {
        vk::MemoryBarrier {
            src_access_mask: self.src_access.legacy_flags(),
            dst_access_mask: self.dst_access.legacy_flags(),
            ..Default::default()
        }
    }

    fn to_buffer_barrier(&self) -> vk::BufferMemoryBarrier<'_> {
        vk::BufferMemoryBarrier {
            src_access_mask: self.src_access.legacy_flags(),
            dst_access_mask: self.dst_access.legacy_flags(),
            src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
            dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
            buffer: self.buffer,
            offset: self.offset,
            size: self.size,
            ..Default::default()
        }
    }

    fn to_image_barrier(&self) -> vk::ImageMemoryBarrier<'_> {
        vk::ImageMemoryBarrier {
            src_access_mask: self.src_access.legacy_flags(),
            dst_access_mask: self.dst_access.legacy_flags(),
            old_layout: vk::ImageLayout::UNDEFINED,
            new_layout: vk::ImageLayout::GENERAL,
            src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
            dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
            image: self.handle,
            subresource_range: vk::ImageSubresourceRange {
                aspect_mask: self.aspect,
                base_mip_level: self.base_mip,
                level_count: self.mip_count,
                base_array_layer: self.base_layer,
                layer_count: self.layer_count,
            },
            ..Default::default()
        }
    }

    fn to_sync2(&self) -> vk::MemoryBarrier2<'_> {
        vk::MemoryBarrier2 {
            src_stage_mask: self.src_stage.flags(),
            src_access_mask: self.src_access.flags2(),
            dst_stage_mask: self.dst_stage.flags(),
            dst_access_mask: self.dst_access.flags2(),
            ..Default::default()
        }
    }

    fn to_buffer_sync2(&self) -> vk::BufferMemoryBarrier2<'_> {
        vk::BufferMemoryBarrier2 {
            src_stage_mask: self.src_stage.flags(),
            src_access_mask: self.src_access.flags2(),
            dst_stage_mask: self.dst_stage.flags(),
            dst_access_mask: self.dst_access.flags2(),
            src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
            dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
            buffer: self.buffer,
            offset: self.offset,
            size: self.size,
            ..Default::default()
        }
    }

    fn to_image_sync2(&self) -> vk::ImageMemoryBarrier2<'_> {
        vk::ImageMemoryBarrier2 {
            src_stage_mask: self.src_stage.flags(),
            src_access_mask: self.src_access.flags2(),
            dst_stage_mask: self.dst_stage.flags(),
            dst_access_mask: self.dst_access.flags2(),
            old_layout: vk::ImageLayout::UNDEFINED,
            new_layout: vk::ImageLayout::GENERAL,
            src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
            dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
            image: self.handle,
            subresource_range: vk::ImageSubresourceRange {
                aspect_mask: self.aspect,
                base_mip_level: self.base_mip,
                level_count: self.mip_count,
                base_array_layer: self.base_layer,
                layer_count: self.layer_count,
            },
            ..Default::default()
        }
    }
}

/// Accumulates barriers and emits them in the caller's sync tier.
#[derive(Debug, Default)]
pub struct BarrierEncoder {
    barriers: Vec<Barrier>,
    tier: Option<SyncTier>,
}

impl BarrierEncoder {
    /// An empty encoder.
    pub fn new() -> BarrierEncoder {
        BarrierEncoder::default()
    }

    /// An empty encoder bound to a device's sync tier.
    pub fn for_tier(tier: SyncTier) -> BarrierEncoder {
        BarrierEncoder {
            barriers: Vec::new(),
            tier: Some(tier),
        }
    }

    /// Queue a barrier, dropping it if it changes nothing.
    pub fn push(&mut self, barrier: Barrier) {
        if !barrier.is_noop() {
            self.barriers.push(barrier);
        }
    }

    /// Queue several barriers.
    pub fn extend(&mut self, barriers: impl IntoIterator<Item = Barrier>) {
        for b in barriers {
            self.push(b);
        }
    }

    /// Number of queued barriers.
    pub fn len(&self) -> usize {
        self.barriers.len()
    }

    /// True when nothing is queued.
    pub fn is_empty(&self) -> bool {
        self.barriers.is_empty()
    }

    /// The queued barriers.
    pub fn barriers(&self) -> &[Barrier] {
        &self.barriers
    }

    /// Discard everything queued.
    pub fn clear(&mut self) {
        self.barriers.clear();
    }

    /// The stage mask the whole batch waits on, for a queue submit.
    pub fn src_stage_mask(&self) -> vk::PipelineStageFlags2 {
        self.barriers
            .iter()
            .fold(vk::PipelineStageFlags2::empty(), |acc, b| {
                acc | b.src_stage.flags()
            })
    }

    /// The stage mask the whole batch unblocks, for a queue submit.
    pub fn dst_stage_mask(&self) -> vk::PipelineStageFlags2 {
        self.barriers
            .iter()
            .fold(vk::PipelineStageFlags2::empty(), |acc, b| {
                acc | b.dst_stage.flags()
            })
    }

    /// Record the queued barriers into a command buffer.
    ///
    /// Emits `vkCmdPipelineBarrier2` on the Sync2 tiers and
    /// `vkCmdPipelineBarrier` on Legacy, batching same-kind barriers into one
    /// call so a frame with fifty image transitions is a handful of calls
    /// rather than fifty.
    ///
    /// # Safety
    ///
    /// `device` and `command_buffer` must be live, and the command buffer must
    /// be in the recording state.
    ///
    /// # Panics
    ///
    /// Panics if the encoder was not bound to a tier with
    /// [`BarrierEncoder::for_tier`].
    pub unsafe fn record(&self, device: &ash::Device, command_buffer: vk::CommandBuffer) {
        let tier = self
            .tier
            .expect("barrier encoder was not bound to a sync tier");
        if self.barriers.is_empty() {
            return;
        }

        if tier.supports_barrier2() {
            let memory: Vec<_> = self
                .barriers
                .iter()
                .filter(|b| b.kind == ResourceKind::Memory)
                .map(|b| b.to_sync2())
                .collect();
            let buffers: Vec<_> = self
                .barriers
                .iter()
                .filter(|b| b.kind == ResourceKind::Buffer)
                .map(|b| b.to_buffer_sync2())
                .collect();
            let images: Vec<_> = self
                .barriers
                .iter()
                .filter(|b| b.kind == ResourceKind::Image)
                .map(|b| b.to_image_sync2())
                .collect();

            let info = vk::DependencyInfo {
                dependency_flags: vk::DependencyFlags::empty(),
                memory_barrier_count: memory.len() as u32,
                p_memory_barriers: memory.as_ptr(),
                buffer_memory_barrier_count: buffers.len() as u32,
                p_buffer_memory_barriers: buffers.as_ptr(),
                image_memory_barrier_count: images.len() as u32,
                p_image_memory_barriers: images.as_ptr(),
                ..Default::default()
            };
            unsafe {
                device.cmd_pipeline_barrier2(command_buffer, &info);
            }
        } else {
            let src = self
                .barriers
                .iter()
                .fold(vk::PipelineStageFlags::empty(), |acc, b| {
                    acc | b.src_stage.legacy_flags()
                });
            let dst = self
                .barriers
                .iter()
                .fold(vk::PipelineStageFlags::empty(), |acc, b| {
                    acc | b.dst_stage.legacy_flags()
                });

            let memory: Vec<_> = self
                .barriers
                .iter()
                .filter(|b| b.kind == ResourceKind::Memory)
                .map(|b| b.to_memory_barrier())
                .collect();
            let buffers: Vec<_> = self
                .barriers
                .iter()
                .filter(|b| b.kind == ResourceKind::Buffer)
                .map(|b| b.to_buffer_barrier())
                .collect();
            let images: Vec<_> = self
                .barriers
                .iter()
                .filter(|b| b.kind == ResourceKind::Image)
                .map(|b| b.to_image_barrier())
                .collect();

            unsafe {
                device.cmd_pipeline_barrier(
                    command_buffer,
                    src,
                    dst,
                    vk::DependencyFlags::empty(),
                    &memory,
                    &buffers,
                    &images,
                )
            };
        }
    }
}

/// The standard barrier set for a linear-memory upload: host writes a staging
/// buffer, then the transfer shader reads it.
pub fn upload_barriers() -> [Barrier; 1] {
    // Make the host write visible to the transfer stage that reads it. Nothing
    // is needed on the far side: the transfer write carries itself.
    [Barrier::memory(
        Stage::None,
        Access::HostWrite,
        Stage::Transfer,
        Access::TransferRead,
    )]
}

/// The barrier that makes a written image readable by a fragment shader.
pub fn image_to_shader_read(image: vk::Image, aspect: vk::ImageAspectFlags) -> Barrier {
    Barrier::image(image, aspect)
        .src(Stage::ColorAttachmentOutput, Access::ColorAttachmentWrite)
        .dst(Stage::FragmentShader, Access::ShaderRead)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ash::vk::Handle as _;

    fn dummy_image() -> vk::Image {
        // A non-null placeholder; the encoder never dereferences it.
        vk::Image::from_raw(0x1234_5678_9abc_def0)
    }

    fn dummy_buffer() -> vk::Buffer {
        vk::Buffer::from_raw(0x1234_5678_9abc_def1)
    }

    #[test]
    fn every_stage_maps_to_a_nonzero_flag_except_none() {
        for stage in [
            Stage::TopOfPipe,
            Stage::VertexInput,
            Stage::VertexShader,
            Stage::FragmentShader,
            Stage::EarlyFragmentTests,
            Stage::LateFragmentTests,
            Stage::ColorAttachmentOutput,
            Stage::ComputeShader,
            Stage::Transfer,
            Stage::BottomOfPipe,
            Stage::AllCommands,
        ] {
            assert!(
                !stage.flags().is_empty(),
                "{stage:?} must map to a stage bit"
            );
        }
        assert!(Stage::None.flags().is_empty());
    }

    #[test]
    fn every_access_maps_to_a_nonzero_flag_except_none() {
        for access in [
            Access::IndirectCommandRead,
            Access::IndexRead,
            Access::VertexAttributeRead,
            Access::UniformRead,
            Access::ShaderRead,
            Access::ShaderWrite,
            Access::ColorAttachmentRead,
            Access::ColorAttachmentWrite,
            Access::DepthStencilAttachmentRead,
            Access::DepthStencilAttachmentWrite,
            Access::TransferRead,
            Access::TransferWrite,
            Access::HostRead,
            Access::HostWrite,
            Access::MemoryRead,
            Access::MemoryWrite,
        ] {
            assert!(
                !access.flags2().is_empty(),
                "{access:?} must map to an access bit"
            );
        }
        assert!(Access::None.flags2().is_empty());
    }

    #[test]
    fn legacy_flags_are_populated_for_every_stage() {
        for stage in [
            Stage::TopOfPipe,
            Stage::VertexShader,
            Stage::ComputeShader,
            Stage::Transfer,
            Stage::AllCommands,
        ] {
            assert!(!stage.legacy_flags().is_empty(), "{stage:?}");
        }
    }

    #[test]
    fn fragment_stages_map_to_top_of_pipe_on_legacy() {
        // 1.0 has no fragment stages, so those must degrade to something valid.
        for stage in [
            Stage::FragmentShader,
            Stage::EarlyFragmentTests,
            Stage::LateFragmentTests,
            Stage::ColorAttachmentOutput,
        ] {
            assert_eq!(
                stage.legacy_flags(),
                vk::PipelineStageFlags::TOP_OF_PIPE,
                "{stage:?}"
            );
        }
    }

    #[test]
    fn legacy_access_is_populated() {
        assert_eq!(
            Access::ShaderRead.legacy_flags(),
            vk::AccessFlags::SHADER_READ
        );
        assert_eq!(
            Access::HostWrite.legacy_flags(),
            vk::AccessFlags::HOST_WRITE
        );
        assert!(Access::None.legacy_flags().is_empty());
    }

    #[test]
    fn a_barrier_with_no_stages_is_a_noop_and_is_dropped() {
        let mut enc = BarrierEncoder::for_tier(SyncTier::Sync2Timeline);
        enc.push(Barrier::memory(
            Stage::None,
            Access::None,
            Stage::None,
            Access::None,
        ));
        assert!(enc.is_empty(), "an empty barrier must not be recorded");
    }

    #[test]
    fn barriers_accumulate() {
        let mut enc = BarrierEncoder::for_tier(SyncTier::Sync2);
        enc.push(
            Barrier::image(dummy_image(), vk::ImageAspectFlags::COLOR)
                .src(Stage::ColorAttachmentOutput, Access::ColorAttachmentWrite)
                .dst(Stage::FragmentShader, Access::ShaderRead),
        );
        enc.push(
            Barrier::buffer(dummy_buffer())
                .src(Stage::Transfer, Access::TransferWrite)
                .dst(Stage::VertexShader, Access::ShaderRead),
        );
        assert_eq!(enc.len(), 2);
        assert!(!enc.is_empty());
    }

    #[test]
    fn clear_empties_the_encoder() {
        let mut enc = BarrierEncoder::for_tier(SyncTier::Legacy);
        enc.push(Barrier::memory(
            Stage::Transfer,
            Access::TransferWrite,
            Stage::None,
            Access::None,
        ));
        assert_eq!(enc.len(), 1);
        enc.clear();
        assert!(enc.is_empty());
    }

    #[test]
    fn extend_pushes_several() {
        let mut enc = BarrierEncoder::for_tier(SyncTier::Sync2);
        enc.extend([
            Barrier::memory(
                Stage::Transfer,
                Access::TransferWrite,
                Stage::None,
                Access::None,
            ),
            Barrier::memory(
                Stage::None,
                Access::None,
                Stage::ComputeShader,
                Access::ShaderWrite,
            ),
        ]);
        assert_eq!(enc.len(), 2);
    }

    #[test]
    fn stage_masks_union_across_the_batch() {
        let mut enc = BarrierEncoder::for_tier(SyncTier::Sync2Timeline);
        enc.push(Barrier::memory(
            Stage::Transfer,
            Access::TransferWrite,
            Stage::FragmentShader,
            Access::ShaderRead,
        ));
        enc.push(Barrier::memory(
            Stage::ComputeShader,
            Access::ShaderWrite,
            Stage::None,
            Access::None,
        ));

        let src = enc.src_stage_mask();
        assert!(src.contains(vk::PipelineStageFlags2::TRANSFER));
        assert!(src.contains(vk::PipelineStageFlags2::COMPUTE_SHADER));

        let dst = enc.dst_stage_mask();
        assert!(dst.contains(vk::PipelineStageFlags2::FRAGMENT_SHADER));
    }

    #[test]
    fn empty_encoder_has_empty_stage_masks() {
        let enc = BarrierEncoder::for_tier(SyncTier::Sync2);
        assert!(enc.src_stage_mask().is_empty());
        assert!(enc.dst_stage_mask().is_empty());
    }

    #[test]
    fn builder_sets_stages_and_access() {
        let b = Barrier::memory(
            Stage::ComputeShader,
            Access::ShaderWrite,
            Stage::Transfer,
            Access::TransferRead,
        );
        assert_eq!(b.src_stage, Stage::ComputeShader);
        assert_eq!(b.dst_access, Access::TransferRead);
    }

    #[test]
    fn buffer_barrier_covers_the_whole_buffer_by_default() {
        let b = Barrier::buffer(dummy_buffer());
        assert_eq!(b.kind, ResourceKind::Buffer);
        assert_eq!(b.size, vk::WHOLE_SIZE);
        assert_eq!(b.buffer, dummy_buffer());
    }

    #[test]
    fn image_barrier_targets_one_mip_and_layer() {
        let b = Barrier::image(dummy_image(), vk::ImageAspectFlags::DEPTH);
        assert_eq!(b.kind, ResourceKind::Image);
        assert_eq!(b.mip_count, 1);
        assert_eq!(b.layer_count, 1);
        assert_eq!(b.aspect, vk::ImageAspectFlags::DEPTH);
        assert!(b.is_noop(), "a fresh image barrier has no transitions yet");
    }

    #[test]
    fn image_to_shader_read_sets_both_sides() {
        let b = image_to_shader_read(dummy_image(), vk::ImageAspectFlags::COLOR);
        assert_eq!(b.src_access, Access::ColorAttachmentWrite);
        assert_eq!(b.dst_access, Access::ShaderRead);
        assert!(!b.is_noop());
    }

    #[test]
    fn upload_barrier_makes_host_writes_visible_to_transfer() {
        let b = upload_barriers()[0];
        assert_eq!(b.src_access, Access::HostWrite);
        assert_eq!(b.dst_access, Access::TransferRead);
        assert_eq!(b.dst_stage, Stage::Transfer);
    }

    #[test]
    fn sync2_conversion_populates_stage_and_access() {
        let b = Barrier::memory(
            Stage::ComputeShader,
            Access::ShaderWrite,
            Stage::FragmentShader,
            Access::ShaderRead,
        );
        let m = b.to_sync2();
        assert_eq!(m.src_stage_mask, vk::PipelineStageFlags2::COMPUTE_SHADER);
        assert_eq!(m.src_access_mask, vk::AccessFlags2::SHADER_WRITE);
        assert_eq!(m.dst_stage_mask, vk::PipelineStageFlags2::FRAGMENT_SHADER);
        assert_eq!(m.dst_access_mask, vk::AccessFlags2::SHADER_READ);
    }

    #[test]
    fn legacy_conversion_populates_stage_and_access() {
        let b = Barrier::memory(
            Stage::ComputeShader,
            Access::ShaderWrite,
            Stage::FragmentShader,
            Access::ShaderRead,
        );
        let m = b.to_memory_barrier();
        assert_eq!(m.src_access_mask, vk::AccessFlags::SHADER_WRITE);
        assert_eq!(m.dst_access_mask, vk::AccessFlags::SHADER_READ);
    }

    #[test]
    fn buffer_conversion_keeps_range_and_handle() {
        let b = Barrier::buffer(dummy_buffer())
            .src(Stage::Transfer, Access::TransferWrite)
            .dst(Stage::VertexShader, Access::ShaderRead);
        let legacy = b.to_buffer_barrier();
        assert_eq!(legacy.buffer, dummy_buffer());
        assert_eq!(legacy.size, vk::WHOLE_SIZE);
        assert_eq!(legacy.src_queue_family_index, vk::QUEUE_FAMILY_IGNORED);

        let sync2 = b.to_buffer_sync2();
        assert_eq!(sync2.buffer, dummy_buffer());
        assert_eq!(sync2.src_stage_mask, vk::PipelineStageFlags2::TRANSFER);
    }

    #[test]
    fn image_conversion_keeps_subresource_range() {
        let b = Barrier::image(dummy_image(), vk::ImageAspectFlags::COLOR)
            .src(Stage::ColorAttachmentOutput, Access::ColorAttachmentWrite)
            .dst(Stage::FragmentShader, Access::ShaderRead);

        let legacy = b.to_image_barrier();
        assert_eq!(
            legacy.subresource_range.aspect_mask,
            vk::ImageAspectFlags::COLOR
        );
        assert_eq!(legacy.subresource_range.level_count, 1);
        assert_eq!(legacy.subresource_range.layer_count, 1);
        assert_eq!(legacy.image, dummy_image());

        let sync2 = b.to_image_sync2();
        assert_eq!(
            sync2.subresource_range.aspect_mask,
            vk::ImageAspectFlags::COLOR
        );
        assert_eq!(
            sync2.src_stage_mask,
            vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT
        );
    }

    #[test]
    fn encoder_can_be_built_for_every_tier() {
        for tier in [SyncTier::Legacy, SyncTier::Sync2, SyncTier::Sync2Timeline] {
            let enc = BarrierEncoder::for_tier(tier);
            assert!(enc.is_empty());
        }
    }

    #[test]
    fn encoder_without_a_tier_defaults_to_none() {
        let enc = BarrierEncoder::new();
        assert!(enc.is_empty());
    }

    #[test]
    fn barriers_are_kept_in_order() {
        let mut enc = BarrierEncoder::for_tier(SyncTier::Sync2);
        enc.push(Barrier::memory(
            Stage::Transfer,
            Access::TransferWrite,
            Stage::None,
            Access::None,
        ));
        enc.push(Barrier::memory(
            Stage::ComputeShader,
            Access::ShaderWrite,
            Stage::None,
            Access::None,
        ));
        let stages: Vec<_> = enc.barriers().iter().map(|b| b.src_stage).collect();
        assert_eq!(stages, vec![Stage::Transfer, Stage::ComputeShader]);
    }
}
