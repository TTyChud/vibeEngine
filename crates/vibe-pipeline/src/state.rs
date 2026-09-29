//! Fixed-function pipeline state.
//!
//! Kept as plain descriptions rather than Vulkan structs so a pipeline can be
//! described, compared and hashed without a device, and so a cache can tell
//! whether a change actually requires rebuilding the pipeline.

use ash::vk;

/// How fragments are combined with what is already in the target.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BlendState {
    /// The source replaces the destination. Correct for an opaque pass.
    #[default]
    Opaque,
    /// Standard source-alpha blending.
    AlphaBlend,
    /// Additive blending, for particles and glows.
    Additive,
    /// Multiply blending, for decals and shadow tints.
    Multiply,
}

impl BlendState {
    /// The Vulkan blend factors and operations.
    pub fn factors(self) -> (vk::BlendFactor, vk::BlendFactor, vk::BlendOp, vk::BlendOp) {
        match self {
            BlendState::Opaque => (
                vk::BlendFactor::ONE,
                vk::BlendFactor::ZERO,
                vk::BlendOp::ADD,
                vk::BlendOp::ADD,
            ),
            BlendState::AlphaBlend => (
                vk::BlendFactor::SRC_ALPHA,
                vk::BlendFactor::ONE_MINUS_SRC_ALPHA,
                vk::BlendOp::ADD,
                vk::BlendOp::ADD,
            ),
            BlendState::Additive => (
                vk::BlendFactor::SRC_ALPHA,
                vk::BlendFactor::ONE,
                vk::BlendOp::ADD,
                vk::BlendOp::ADD,
            ),
            BlendState::Multiply => (
                vk::BlendFactor::DST_COLOR,
                vk::BlendFactor::ZERO,
                vk::BlendOp::ADD,
                vk::BlendOp::ADD,
            ),
        }
    }

    /// True when this state writes every channel.
    pub fn is_opaque(self) -> bool {
        self == BlendState::Opaque
    }
}

/// How depth is tested and written.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct DepthState {
    /// Whether depth is tested at all.
    pub enabled: bool,
    /// Whether passing fragments write depth.
    pub write: bool,
    /// The comparison for a fragment that has depth.
    pub compare: DepthCompare,
}

/// The depth comparison function.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DepthCompare {
    /// Never passes, so depth is write-only.
    Never,
    #[default]
    Less,
    LessOrEqual,
    Equal,
    Greater,
    GreaterOrEqual,
    NotEqual,
    Always,
}

impl DepthCompare {
    /// The Vulkan comparison op.
    pub const fn vk(self) -> vk::CompareOp {
        match self {
            DepthCompare::Never => vk::CompareOp::NEVER,
            DepthCompare::Less => vk::CompareOp::LESS,
            DepthCompare::LessOrEqual => vk::CompareOp::LESS_OR_EQUAL,
            DepthCompare::Equal => vk::CompareOp::EQUAL,
            DepthCompare::Greater => vk::CompareOp::GREATER,
            DepthCompare::GreaterOrEqual => vk::CompareOp::GREATER_OR_EQUAL,
            DepthCompare::NotEqual => vk::CompareOp::NOT_EQUAL,
            DepthCompare::Always => vk::CompareOp::ALWAYS,
        }
    }
}

impl DepthState {
    /// Depth testing on, writing, with the usual less comparison.
    pub fn test_and_write() -> DepthState {
        DepthState {
            enabled: true,
            write: true,
            compare: DepthCompare::Less,
        }
    }

    /// Depth testing on, not writing, for a pass that only reads depth.
    pub fn test_only() -> DepthState {
        DepthState {
            enabled: true,
            write: false,
            compare: DepthCompare::Less,
        }
    }

    /// No depth at all, for a 2D overlay drawn last.
    pub fn disabled() -> DepthState {
        DepthState {
            enabled: false,
            write: false,
            compare: DepthCompare::Always,
        }
    }

    /// The Vulkan depth stencil state.
    pub fn vk(self) -> vk::PipelineDepthStencilStateCreateInfo<'static> {
        vk::PipelineDepthStencilStateCreateInfo {
            depth_test_enable: if self.enabled { vk::TRUE } else { vk::FALSE },
            depth_write_enable: if self.write { vk::TRUE } else { vk::FALSE },
            depth_compare_op: self.compare.vk(),
            ..Default::default()
        }
    }
}

/// How triangles are assembled and filled.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct PrimitiveState {
    /// The topology.
    pub topology: Topology,
    /// Which face is front-facing.
    pub cull_mode: CullMode,
    /// Whether to draw both faces.
    pub front_face: FrontFace,
}

/// The primitive topology.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Topology {
    /// Every three vertices is a triangle.
    #[default]
    Triangles,
    /// Independent lines.
    Lines,
    /// A connected line strip.
    LineStrip,
    /// Independent points.
    Points,
}

impl Topology {
    /// The Vulkan topology.
    pub const fn vk(self) -> vk::PrimitiveTopology {
        match self {
            Topology::Triangles => vk::PrimitiveTopology::TRIANGLE_LIST,
            Topology::Lines => vk::PrimitiveTopology::LINE_LIST,
            Topology::LineStrip => vk::PrimitiveTopology::LINE_STRIP,
            Topology::Points => vk::PrimitiveTopology::POINT_LIST,
        }
    }
}

/// Which triangle faces are discarded.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CullMode {
    /// Draw both faces, which is what a 2D sprite pass wants.
    #[default]
    None,
    CullBack,
    CullFront,
    CullBoth,
}

impl CullMode {
    /// The Vulkan cull mode.
    pub const fn vk(self) -> vk::CullModeFlags {
        match self {
            CullMode::None => vk::CullModeFlags::NONE,
            CullMode::CullBack => vk::CullModeFlags::BACK,
            CullMode::CullFront => vk::CullModeFlags::FRONT,
            CullMode::CullBoth => vk::CullModeFlags::FRONT_AND_BACK,
        }
    }
}

/// Which winding is front-facing.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum FrontFace {
    #[default]
    CounterClockwise,
    Clockwise,
}

impl FrontFace {
    /// The Vulkan front face.
    pub const fn vk(self) -> vk::FrontFace {
        match self {
            FrontFace::CounterClockwise => vk::FrontFace::COUNTER_CLOCKWISE,
            FrontFace::Clockwise => vk::FrontFace::CLOCKWISE,
        }
    }
}

/// Rasterisation state.
///
/// Vulkan 1.3 removed `depthClipEnable` and `depthBoundsEnable` from the
/// pipeline state: both are now always on, so there is nothing to configure
/// here. A pipeline that needs clipping behaviour expresses it by discarding
/// fragments or running a different pipeline, not by toggling a flag.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RasterState {
    /// Whether the rasteriser runs at all; when true nothing is rasterised.
    pub rasterizer_discard_enable: bool,
    /// Line width; only 1.0 is guaranteed on every implementation.
    pub line_width: f32,
    /// Depth bias constant factor, for shadow and decal passes.
    pub depth_bias_constant: f32,
    /// Depth bias slope factor, scaled by the primitive's slope.
    pub depth_bias_slope: f32,
    /// Whether depth bias is applied at all.
    pub depth_bias_enable: bool,
}

impl Default for RasterState {
    fn default() -> Self {
        RasterState {
            rasterizer_discard_enable: false,
            line_width: 1.0,
            depth_bias_constant: 0.0,
            depth_bias_slope: 0.0,
            depth_bias_enable: false,
        }
    }
}

impl RasterState {
    /// The Vulkan rasterisation state.
    pub fn vk(self) -> vk::PipelineRasterizationStateCreateInfo<'static> {
        vk::PipelineRasterizationStateCreateInfo {
            rasterizer_discard_enable: if self.rasterizer_discard_enable {
                vk::TRUE
            } else {
                vk::FALSE
            },
            depth_bias_enable: if self.depth_bias_enable {
                vk::TRUE
            } else {
                vk::FALSE
            },
            depth_bias_constant_factor: self.depth_bias_constant,
            depth_bias_slope_factor: self.depth_bias_slope,
            line_width: self.line_width,
            ..Default::default()
        }
    }
}

/// Multisampling state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct SampleState {
    /// Samples per pixel; 1 disables multisampling.
    pub raster_samples: vk::SampleCountFlags,
}

impl SampleState {
    /// One sample, which is what a 2D batch renderer wants.
    pub fn single() -> SampleState {
        SampleState {
            raster_samples: vk::SampleCountFlags::TYPE_1,
        }
    }

    /// Four samples, for edges that need them.
    pub fn four() -> SampleState {
        SampleState {
            raster_samples: vk::SampleCountFlags::TYPE_4,
        }
    }

    /// True when multisampling is on.
    pub fn is_multisampled(self) -> bool {
        self.raster_samples != vk::SampleCountFlags::TYPE_1
    }
}

/// A whole pipeline description, independent of any device.
#[derive(Debug, Clone, PartialEq)]
pub struct PipelineDesc {
    /// Human-facing name, for caches and error messages.
    pub name: String,
    /// The vertex input layout.
    pub layout: crate::layout::VertexLayout,
    /// Compiled vertex SPIR-V, or `None` for a compute pipeline.
    pub vertex_spirv: Option<Vec<u32>>,
    /// Compiled fragment SPIR-V, or `None`.
    pub fragment_spirv: Option<Vec<u32>>,
    /// Compiled compute SPIR-V, for a compute pipeline.
    pub compute_spirv: Option<Vec<u32>>,
    /// Push constant range in bytes, 0 for none.
    pub push_constant_bytes: u32,
    /// Colour blending.
    pub blend: BlendState,
    /// Depth state.
    pub depth: DepthState,
    /// Primitive assembly.
    pub primitive: PrimitiveState,
    /// Rasterisation.
    pub raster: RasterState,
    /// Multisampling.
    pub samples: SampleState,
    /// Number of colour attachments the pipeline writes.
    pub color_attachment_count: u32,
}

impl Default for PipelineDesc {
    fn default() -> Self {
        PipelineDesc {
            name: String::new(),
            layout: crate::layout::vertex_layout_for_quad(),
            vertex_spirv: None,
            fragment_spirv: None,
            compute_spirv: None,
            push_constant_bytes: 0,
            blend: BlendState::default(),
            depth: DepthState::default(),
            primitive: PrimitiveState::default(),
            raster: RasterState::default(),
            samples: SampleState::single(),
            color_attachment_count: 1,
        }
    }
}

impl PipelineDesc {
    /// A description for the engine's 2D batched quad.
    pub fn quad_2d(name: impl Into<String>) -> PipelineDesc {
        PipelineDesc {
            name: name.into(),
            layout: crate::layout::vertex_layout_for_quad(),
            // A 2D camera is a single mat4, which is 64 bytes of push constants.
            push_constant_bytes: 64,
            // 2D sprites are drawn in back-to-front order by the batcher, so
            // depth is off entirely rather than sorted.
            depth: DepthState::disabled(),
            ..Default::default()
        }
    }

    /// A description for a compute pass.
    pub fn compute(name: impl Into<String>) -> PipelineDesc {
        PipelineDesc {
            name: name.into(),
            layout: crate::layout::VertexLayout::new(0, vec![]),
            color_attachment_count: 0,
            ..Default::default()
        }
    }

    /// True when this describes a compute pipeline.
    pub fn is_compute(&self) -> bool {
        self.compute_spirv.is_some() && self.vertex_spirv.is_none()
    }

    /// Check the description is coherent.
    pub fn validate(&self) -> Result<(), crate::error::PipelineError> {
        if self.is_compute() {
            if self.compute_spirv.is_none() {
                return Err(crate::error::PipelineError::NoStages);
            }
            return Ok(());
        }
        if self.vertex_spirv.is_none() {
            return Err(crate::error::PipelineError::NoStages);
        }
        if !self.is_compute() {
            self.layout.validate()?;
        }
        Ok(())
    }

    /// A stable key for pipeline caching.
    ///
    /// Two descriptions with the same key produce an identical pipeline, so a
    /// cache can skip the driver call entirely.
    pub fn cache_key(&self) -> u64 {
        let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
        let mut mix = |bytes: &[u8]| {
            for b in bytes {
                hash ^= *b as u64;
                hash = hash.wrapping_mul(0x100_0000_01b3);
            }
        };
        mix(self.name.as_bytes());
        mix(&self.layout.stride.to_le_bytes());
        for a in &self.layout.attributes {
            mix(&a.location.to_le_bytes());
            mix(&a.offset.to_le_bytes());
            mix(&(a.format.size() as u32).to_le_bytes());
        }
        for stage in [
            &self.vertex_spirv,
            &self.fragment_spirv,
            &self.compute_spirv,
        ] {
            match stage {
                Some(words) => {
                    mix(&[1]);
                    mix(&(words.len() as u32).to_le_bytes());
                }
                None => mix(&[0]),
            }
        }
        mix(&self.push_constant_bytes.to_le_bytes());
        mix(&[
            self.blend as u8,
            self.depth.enabled as u8,
            self.depth.write as u8,
        ]);
        mix(&[self.depth.compare as u8]);
        mix(&[
            self.primitive.topology as u8,
            self.primitive.cull_mode as u8,
        ]);
        mix(&self.color_attachment_count.to_le_bytes());
        mix(&[self.samples.raster_samples.as_raw() as u8]);
        hash
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn opaque_blend_replaces_the_target() {
        let (src, dst, op, _) = BlendState::Opaque.factors();
        assert_eq!(src, vk::BlendFactor::ONE);
        assert_eq!(dst, vk::BlendFactor::ZERO);
        assert_eq!(op, vk::BlendOp::ADD);
    }

    #[test]
    fn alpha_blend_uses_the_standard_factors() {
        let (src, dst, _, _) = BlendState::AlphaBlend.factors();
        assert_eq!(src, vk::BlendFactor::SRC_ALPHA);
        assert_eq!(dst, vk::BlendFactor::ONE_MINUS_SRC_ALPHA);
    }

    #[test]
    fn additive_blend_saturates() {
        let (_, dst, _, _) = BlendState::Additive.factors();
        assert_eq!(dst, vk::BlendFactor::ONE);
    }

    #[test]
    fn multiply_blend_scales_by_the_target() {
        let (src, dst, _, _) = BlendState::Multiply.factors();
        assert_eq!(src, vk::BlendFactor::DST_COLOR);
        assert_eq!(dst, vk::BlendFactor::ZERO);
    }

    #[test]
    fn only_opaque_counts_as_opaque() {
        assert!(BlendState::Opaque.is_opaque());
        assert!(!BlendState::AlphaBlend.is_opaque());
    }

    #[test]
    fn depth_disabled_writes_nothing() {
        let d = DepthState::disabled();
        let vk = d.vk();
        assert_eq!(vk.depth_test_enable, vk::FALSE);
        assert_eq!(vk.depth_write_enable, vk::FALSE);
    }

    #[test]
    fn depth_test_only_does_not_write() {
        let d = DepthState::test_only();
        let vk = d.vk();
        assert_eq!(vk.depth_test_enable, vk::TRUE);
        assert_eq!(vk.depth_write_enable, vk::FALSE);
    }

    #[test]
    fn depth_test_and_write_both_enable() {
        let vk = DepthState::test_and_write().vk();
        assert_eq!(vk.depth_test_enable, vk::TRUE);
        assert_eq!(vk.depth_write_enable, vk::TRUE);
        assert_eq!(vk.depth_compare_op, vk::CompareOp::LESS);
    }

    #[test]
    fn every_depth_compare_maps_to_a_vulkan_op() {
        for (ours, theirs) in [
            (DepthCompare::Never, vk::CompareOp::NEVER),
            (DepthCompare::Less, vk::CompareOp::LESS),
            (DepthCompare::LessOrEqual, vk::CompareOp::LESS_OR_EQUAL),
            (DepthCompare::Equal, vk::CompareOp::EQUAL),
            (DepthCompare::Greater, vk::CompareOp::GREATER),
            (
                DepthCompare::GreaterOrEqual,
                vk::CompareOp::GREATER_OR_EQUAL,
            ),
            (DepthCompare::NotEqual, vk::CompareOp::NOT_EQUAL),
            (DepthCompare::Always, vk::CompareOp::ALWAYS),
        ] {
            assert_eq!(ours.vk(), theirs, "{ours:?}");
        }
    }

    #[test]
    fn topology_maps_to_vulkan() {
        assert_eq!(
            Topology::Triangles.vk(),
            vk::PrimitiveTopology::TRIANGLE_LIST
        );
        assert_eq!(Topology::Lines.vk(), vk::PrimitiveTopology::LINE_LIST);
        assert_eq!(Topology::LineStrip.vk(), vk::PrimitiveTopology::LINE_STRIP);
        assert_eq!(Topology::Points.vk(), vk::PrimitiveTopology::POINT_LIST);
    }

    #[test]
    fn cull_modes_map_to_vulkan() {
        assert_eq!(CullMode::None.vk(), vk::CullModeFlags::NONE);
        assert_eq!(CullMode::CullBack.vk(), vk::CullModeFlags::BACK);
        assert_eq!(CullMode::CullFront.vk(), vk::CullModeFlags::FRONT);
        assert_eq!(CullMode::CullBoth.vk(), vk::CullModeFlags::FRONT_AND_BACK);
    }

    #[test]
    fn a_2d_pipeline_culls_nothing() {
        assert_eq!(PrimitiveState::default().cull_mode, CullMode::None);
    }

    #[test]
    fn front_face_maps_to_vulkan() {
        assert_eq!(
            FrontFace::CounterClockwise.vk(),
            vk::FrontFace::COUNTER_CLOCKWISE
        );
        assert_eq!(FrontFace::Clockwise.vk(), vk::FrontFace::CLOCKWISE);
    }

    #[test]
    fn raster_defaults_draw_nothing_special() {
        let r = RasterState::default();
        assert!(!r.rasterizer_discard_enable);
        assert!(!r.depth_bias_enable);
        assert_eq!(r.line_width, 1.0);
        assert_eq!(r.vk().rasterizer_discard_enable, vk::FALSE);
        assert_eq!(r.vk().line_width, 1.0);
    }

    #[test]
    fn depth_bias_reaches_the_pipeline_state() {
        let r = RasterState {
            depth_bias_enable: true,
            depth_bias_constant: 2.0,
            depth_bias_slope: 4.0,
            ..Default::default()
        };
        let vk = r.vk();
        assert_eq!(vk.depth_bias_enable, vk::TRUE);
        assert_eq!(vk.depth_bias_constant_factor, 2.0);
        assert_eq!(vk.depth_bias_slope_factor, 4.0);
    }

    #[test]
    fn discarding_rasterisation_is_reported() {
        let r = RasterState {
            rasterizer_discard_enable: true,
            ..Default::default()
        };
        assert_eq!(r.vk().rasterizer_discard_enable, vk::TRUE);
    }

    #[test]
    fn single_sampling_is_not_multisampled() {
        assert!(!SampleState::single().is_multisampled());
        assert!(SampleState::four().is_multisampled());
        assert_eq!(
            SampleState::single().raster_samples,
            vk::SampleCountFlags::TYPE_1
        );
    }

    #[test]
    fn the_quad_pipeline_uses_the_quad_layout() {
        let p = PipelineDesc::quad_2d("quad");
        assert_eq!(p.layout, crate::layout::vertex_layout_for_quad());
        assert_eq!(p.color_attachment_count, 1);
    }

    #[test]
    fn the_quad_pipeline_has_depth_disabled() {
        assert!(!PipelineDesc::quad_2d("quad").depth.enabled);
    }

    #[test]
    fn the_quad_pipeline_reserves_a_mat4_of_push_constants() {
        assert_eq!(PipelineDesc::quad_2d("quad").push_constant_bytes, 64);
    }

    #[test]
    fn a_compute_pipeline_has_no_colour_targets() {
        let p = PipelineDesc::compute("cull");
        assert_eq!(p.color_attachment_count, 0);
    }

    #[test]
    fn a_pipeline_with_no_stages_is_rejected() {
        let p = PipelineDesc::default();
        assert!(matches!(
            p.validate(),
            Err(crate::error::PipelineError::NoStages)
        ));
    }

    #[test]
    fn a_graphics_pipeline_with_only_a_vertex_stage_is_valid() {
        let p = PipelineDesc {
            vertex_spirv: Some(vec![1, 2, 3]),
            ..PipelineDesc::quad_2d("q")
        };
        assert!(p.validate().is_ok());
    }

    #[test]
    fn a_compute_pipeline_with_spirv_is_valid() {
        let p = PipelineDesc {
            compute_spirv: Some(vec![1]),
            ..PipelineDesc::compute("c")
        };
        assert!(p.is_compute());
        assert!(p.validate().is_ok());
    }

    #[test]
    fn an_invalid_layout_fails_pipeline_validation() {
        let p = PipelineDesc {
            vertex_spirv: Some(vec![1]),
            layout: crate::layout::VertexLayout::new(
                8,
                vec![crate::layout::Attribute::new(
                    0,
                    0,
                    crate::layout::VertexFormat::Float4,
                )],
            ),
            ..PipelineDesc::quad_2d("q")
        };
        assert!(p.validate().is_err());
    }

    #[test]
    fn identical_descriptions_share_a_cache_key() {
        let a = PipelineDesc {
            vertex_spirv: Some(vec![1, 2]),
            ..PipelineDesc::quad_2d("q")
        };
        let b = a.clone();
        assert_eq!(a.cache_key(), b.cache_key());
    }

    #[test]
    fn a_different_name_changes_the_cache_key() {
        let a = PipelineDesc {
            vertex_spirv: Some(vec![1]),
            ..PipelineDesc::quad_2d("a")
        };
        let b = PipelineDesc {
            vertex_spirv: Some(vec![1]),
            ..PipelineDesc::quad_2d("b")
        };
        assert_ne!(a.cache_key(), b.cache_key());
    }

    #[test]
    fn a_different_blend_changes_the_cache_key() {
        let a = PipelineDesc {
            vertex_spirv: Some(vec![1]),
            ..PipelineDesc::quad_2d("q")
        };
        let b = PipelineDesc {
            vertex_spirv: Some(vec![1]),
            blend: BlendState::AlphaBlend,
            ..PipelineDesc::quad_2d("q")
        };
        assert_ne!(a.cache_key(), b.cache_key());
    }

    #[test]
    fn a_different_depth_state_changes_the_cache_key() {
        let a = PipelineDesc {
            vertex_spirv: Some(vec![1]),
            ..PipelineDesc::quad_2d("q")
        };
        let b = PipelineDesc {
            vertex_spirv: Some(vec![1]),
            depth: DepthState::test_and_write(),
            ..PipelineDesc::quad_2d("q")
        };
        assert_ne!(a.cache_key(), b.cache_key());
    }

    #[test]
    fn a_different_push_constant_size_changes_the_cache_key() {
        let a = PipelineDesc {
            vertex_spirv: Some(vec![1]),
            ..PipelineDesc::quad_2d("q")
        };
        let b = PipelineDesc {
            vertex_spirv: Some(vec![1]),
            push_constant_bytes: 128,
            ..PipelineDesc::quad_2d("q")
        };
        assert_ne!(a.cache_key(), b.cache_key());
    }

    #[test]
    fn a_different_vertex_stride_changes_the_cache_key() {
        let a = PipelineDesc {
            vertex_spirv: Some(vec![1]),
            ..PipelineDesc::quad_2d("q")
        };
        let b = PipelineDesc {
            vertex_spirv: Some(vec![1]),
            layout: crate::layout::VertexLayout::new(64, a.layout.attributes.clone()),
            ..PipelineDesc::quad_2d("q")
        };
        assert_ne!(a.cache_key(), b.cache_key());
    }

    #[test]
    fn a_different_attachment_count_changes_the_cache_key() {
        let a = PipelineDesc {
            vertex_spirv: Some(vec![1]),
            ..PipelineDesc::quad_2d("q")
        };
        let b = PipelineDesc {
            vertex_spirv: Some(vec![1]),
            color_attachment_count: 2,
            ..PipelineDesc::quad_2d("q")
        };
        assert_ne!(a.cache_key(), b.cache_key());
    }

    #[test]
    fn a_different_shader_presence_changes_the_cache_key() {
        let a = PipelineDesc {
            vertex_spirv: Some(vec![1]),
            ..PipelineDesc::quad_2d("q")
        };
        let b = PipelineDesc {
            vertex_spirv: Some(vec![1]),
            fragment_spirv: Some(vec![2]),
            ..PipelineDesc::quad_2d("q")
        };
        assert_ne!(a.cache_key(), b.cache_key());
    }

    #[test]
    fn the_cache_key_is_stable_across_calls() {
        let p = PipelineDesc {
            vertex_spirv: Some(vec![9, 9, 9]),
            ..PipelineDesc::quad_2d("q")
        };
        let first = p.cache_key();
        for _ in 0..5 {
            assert_eq!(p.cache_key(), first);
        }
    }
}
