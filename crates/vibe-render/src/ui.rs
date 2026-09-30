//! The immediate-mode UI pipeline: a triangle list of coloured vertices.
//!
//! The quad batcher submits axis-aligned quads indexed by a shader, which is the
//! right shape for sprites and the wrong shape for an editor: a UI is arbitrary
//! triangles with their own vertices, drawn back to front with alpha blending
//! and no depth. So the UI gets its own pipeline over the same infrastructure —
//! the same barrier encoder, the same descriptor plumbing, the same frame — and
//! differs only in the vertex layout and the topology.
//!
//! The vertex is deliberately plain: position in pixels, a colour, and nothing
//! else. A UI font atlas is a single texture bound once, and every vertex
//! references the same one, so a per-vertex texture index would be 4 bytes of
//! nothing per vertex.

use ash::vk;
use vibe_pipeline::state::Topology;
use vibe_pipeline::{
    Attribute, BlendState, PipelineDesc, PrimitiveState, VertexFormat, VertexLayout,
};

use crate::error::RenderError;
use crate::pipeline::create_module;
use vibe_shader::ShaderCompiler;

/// The vertex the UI pipeline consumes.
///
/// `#[repr(C)]` and 12 bytes: two position floats and four colour bytes. The
/// fields add up exactly, so there is no padding and no hole for a `bytemuck`
/// cast to read. The stride is not padded to a multiple of 16 because the rule
/// that asks for one is about per-vertex fetch granularity on larger attributes,
/// and every field here is 4-byte aligned.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct UiVertex {
    /// Position in pixels, origin top left, matching egui's own space.
    pub position: [f32; 2],
    /// Premultiplied RGBA, each byte 0..255.
    pub color: [u8; 4],
}

/// Bytes one UI vertex occupies.
pub const UI_VERTEX_SIZE: u32 = std::mem::size_of::<UiVertex>() as u32;

/// The vertex layout for [`UiVertex`].
pub fn ui_vertex_layout() -> VertexLayout {
    VertexLayout::new(
        UI_VERTEX_SIZE,
        vec![
            Attribute::new(0, 0, VertexFormat::Float2),
            Attribute::new(1, 8, VertexFormat::Unorm8x4),
        ],
    )
}

/// The UI shader, in WGSL.
///
/// There is no camera uniform: the UI's vertices are already in screen pixels
/// and the pipeline's viewport maps them, so a per-frame matrix would be a
/// uniform upload that changes nothing. The transform is a
/// `pixels_to_clip` push constant instead, which is what converts the top-left
/// origin to Vulkan's clip space, and it is one value rather than 16 floats.
pub const UI_SHADER: &str = r#"
struct UiVertex {
    @location(0) position: vec2<f32>,
    @location(1) color: vec4<f32>,
}

struct VertexOut {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
}

struct Pixels {
    // (2/width, 2/height, -1, -1): the top-left pixel corner to clip (-1, -1).
    scale_offset: vec4<f32>,
}

// A uniform buffer rather than a push constant, because naga 0.30's WGSL front
// end has no push-constant address space: `var<push_constant>` is an unknown
// address space and `@push_constant` is an unknown attribute, so a shader
// written that way does not parse at all. The quad pipeline already takes its
// camera the same way, so this is also the shape the project is used to.
@group(0) @binding(0) var<uniform> transform: Pixels;

@vertex
fn vs_main(vertex: UiVertex) -> VertexOut {
    var out: VertexOut;
    let p = vertex.position;
    out.clip_position = vec4<f32>(
        p.x * transform.scale_offset.x + transform.scale_offset.z,
        p.y * transform.scale_offset.y + transform.scale_offset.w,
        0.0,
        1.0,
    );
    out.color = vertex.color;
    return out;
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    return in.color;
}
"#;

/// The name of the vertex entry point.
pub const VERTEX_ENTRY: &str = "vs_main";
/// The name of the fragment entry point.
pub const FRAGMENT_ENTRY: &str = "fs_main";

/// The descriptor bindings the UI shader uses. It uses none: the transform is
/// a push constant and the fragment stage returns the vertex colour, so the
/// pipeline layout declares no set at all. These are kept because a future
/// textured UI panel would reintroduce them, and because a declared-but-unbound
/// set is what makes a draw invalid rather than merely wasteful.
pub const TRANSFORM_BINDING: u32 = 0;
/// The font atlas texture, for a panel that draws one.
pub const ATLAS_BINDING: u32 = 1;
/// The font atlas sampler, for a panel that draws one.
pub const SAMPLER_BINDING: u32 = 2;

/// The push-constant block the vertex stage reads.
///
/// A 16-byte block, which is the minimum a push constant range can be, and
/// small enough to set with one `cmd_push_constants`.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ScreenTransform {
    /// `(2/width, 2/height, -1, -1)`.
    pub scale_offset: [f32; 4],
}

impl ScreenTransform {
    /// The transform for a surface of the given size.
    ///
    /// A zero or non-finite size gives the identity rather than a division by
    /// zero: a minimised window is reported as a zero extent, and a NaN scale
    /// would make every UI vertex NaN and blank the whole interface for as long
    /// as the window stayed minimised.
    pub fn for_size(width: u32, height: u32) -> ScreenTransform {
        if width == 0 || height == 0 {
            return ScreenTransform::default();
        }
        ScreenTransform {
            scale_offset: [2.0 / width as f32, 2.0 / height as f32, -1.0, -1.0],
        }
    }
}

/// The description the UI pipeline is built from.
pub fn ui_pipeline_desc() -> PipelineDesc {
    PipelineDesc {
        name: "ui".to_string(),
        layout: ui_vertex_layout(),
        // No push constants: naga 0.30's WGSL front end cannot parse them, so
        // the transform is a uniform buffer like the quad pipeline's camera.
        push_constant_bytes: 0,
        // The UI is drawn back to front with alpha, exactly like the 2D
        // batcher, and with no depth so a panel always covers the scene.
        blend: BlendState::AlphaBlend,
        primitive: PrimitiveState {
            topology: Topology::Triangles,
            ..Default::default()
        },
        ..PipelineDesc::quad_2d("ui")
    }
}

/// The dynamic states the UI pipeline enables.
///
/// The scissor is what makes a clipped primitive work: egui hands back each mesh
/// with the rectangle it was drawn into, and without a scissor the parts of a
/// panel that scrolled out of view would still be drawn over the viewport.
pub fn ui_dynamic_states() -> Vec<vk::DynamicState> {
    vec![
        vk::DynamicState::VIEWPORT,
        vk::DynamicState::SCISSOR,
        vk::DynamicState::CULL_MODE,
        vk::DynamicState::PRIMITIVE_TOPOLOGY,
        vk::DynamicState::DEPTH_TEST_ENABLE,
    ]
}

/// A built UI pipeline and the objects it owns.
pub struct UiPipeline {
    /// The pipeline itself.
    pub pipeline: vk::Pipeline,
    /// The layout the pipeline was built against.
    pub layout: vk::PipelineLayout,
    /// The descriptor set layout for the transform.
    pub descriptor_layout: vk::DescriptorSetLayout,
    /// The compiled vertex shader.
    pub vertex_module: vk::ShaderModule,
    /// The compiled fragment shader.
    pub fragment_module: vk::ShaderModule,
    /// The cache key of the description this was built from.
    pub cache_key: u64,
}

impl std::fmt::Debug for UiPipeline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("UiPipeline")
            .field("pipeline", &self.pipeline)
            .field("cache_key", &self.cache_key)
            .finish()
    }
}

/// Compile and build the UI pipeline.
///
/// # Safety
///
/// `device` must be a live logical device.
pub unsafe fn build_ui_pipeline<C: ShaderCompiler>(
    device: &ash::Device,
    compiler: &C,
    target_formats: &[vk::Format],
) -> Result<UiPipeline, RenderError> {
    use vibe_shader::CompileOptions;

    let vertex_spirv = vibe_shader::compile_entry_point(
        UI_SHADER,
        CompileOptions::release(),
        Some((VERTEX_ENTRY, naga::ShaderStage::Vertex)),
    )
    .map_err(|e| RenderError::Pipeline(e.to_string()))?;
    let fragment_spirv = vibe_shader::compile_entry_point(
        UI_SHADER,
        CompileOptions::release(),
        Some((FRAGMENT_ENTRY, naga::ShaderStage::Fragment)),
    )
    .map_err(|e| RenderError::Pipeline(e.to_string()))?;
    // Ask the injected compiler for the module once: a source it cannot handle
    // then fails here rather than at pipeline creation.
    compiler
        .compile(UI_SHADER, CompileOptions::release())
        .map_err(|e| RenderError::Pipeline(e.to_string()))?;

    let desc = PipelineDesc {
        vertex_spirv: Some(vertex_spirv),
        fragment_spirv: Some(fragment_spirv),
        ..ui_pipeline_desc()
    };
    desc.validate()
        .map_err(|e| RenderError::Pipeline(e.to_string()))?;
    let cache_key = desc.cache_key();

    unsafe {
        let vertex_module = create_module(device, desc.vertex_spirv.as_ref().unwrap())?;
        let fragment_module = match create_module(device, desc.fragment_spirv.as_ref().unwrap()) {
            Ok(m) => m,
            Err(e) => {
                device.destroy_shader_module(vertex_module, None);
                return Err(e);
            }
        };

        // Exactly one set, holding the screen transform, which the shader reads
        // at binding 0. Declaring a set the shader does not use is not merely
        // wasteful: Vulkan requires every statically-used set to be bound at
        // draw time, so a declared-but-unbound set makes the draw invalid.
        let bindings = [vk::DescriptorSetLayoutBinding {
            binding: TRANSFORM_BINDING,
            descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
            descriptor_count: 1,
            stage_flags: vk::ShaderStageFlags::VERTEX,
            ..Default::default()
        }];
        let layout_info = vk::DescriptorSetLayoutCreateInfo {
            binding_count: bindings.len() as u32,
            p_bindings: bindings.as_ptr(),
            ..Default::default()
        };
        let descriptor_layout = device.create_descriptor_set_layout(&layout_info, None)?;

        let pipeline_layout_info = vk::PipelineLayoutCreateInfo {
            set_layout_count: 1,
            p_set_layouts: &descriptor_layout,
            push_constant_range_count: 0,
            p_push_constant_ranges: std::ptr::null(),
            ..Default::default()
        };
        let layout = match device.create_pipeline_layout(&pipeline_layout_info, None) {
            Ok(l) => l,
            Err(e) => {
                device.destroy_descriptor_set_layout(descriptor_layout, None);
                device.destroy_shader_module(vertex_module, None);
                device.destroy_shader_module(fragment_module, None);
                return Err(RenderError::Vk(e));
            }
        };

        let stages = [
            vk::PipelineShaderStageCreateInfo {
                stage: vk::ShaderStageFlags::VERTEX,
                module: vertex_module,
                p_name: c"vs_main".as_ptr(),
                ..Default::default()
            },
            vk::PipelineShaderStageCreateInfo {
                stage: vk::ShaderStageFlags::FRAGMENT,
                module: fragment_module,
                p_name: c"fs_main".as_ptr(),
                ..Default::default()
            },
        ];

        let target_formats: Vec<vk::Format> = target_formats.to_vec();
        let attributes = desc.layout.attribute_descriptions();
        let binding = desc.layout.binding_description();
        let vertex_input = vk::PipelineVertexInputStateCreateInfo {
            vertex_binding_description_count: 1,
            p_vertex_binding_descriptions: &binding,
            vertex_attribute_description_count: attributes.len() as u32,
            p_vertex_attribute_descriptions: attributes.as_ptr(),
            ..Default::default()
        };
        let input_assembly = vk::PipelineInputAssemblyStateCreateInfo {
            topology: desc.primitive.topology.vk(),
            primitive_restart_enable: vk::FALSE,
            ..Default::default()
        };
        let dynamic_states = ui_dynamic_states();
        let dynamic = vk::PipelineDynamicStateCreateInfo {
            dynamic_state_count: dynamic_states.len() as u32,
            p_dynamic_states: dynamic_states.as_ptr(),
            ..Default::default()
        };
        let viewport_state = vk::PipelineViewportStateCreateInfo {
            viewport_count: 1,
            scissor_count: 1,
            ..Default::default()
        };
        let rasterization = desc.raster.vk();
        let multisample = vk::PipelineMultisampleStateCreateInfo {
            rasterization_samples: desc.samples.raster_samples,
            ..Default::default()
        };
        let depth_stencil = desc.depth.vk();
        let (src_c, dst_c, op_c) = desc.blend.color_factors();
        let (src_a, dst_a, op_a) = desc.blend.alpha_factors();
        let blend_attachment = vk::PipelineColorBlendAttachmentState {
            blend_enable: if desc.blend.is_opaque() {
                vk::FALSE
            } else {
                vk::TRUE
            },
            src_color_blend_factor: src_c,
            dst_color_blend_factor: dst_c,
            color_blend_op: op_c,
            src_alpha_blend_factor: src_a,
            dst_alpha_blend_factor: dst_a,
            alpha_blend_op: op_a,
            // Every channel: a panel with alpha must still write RGB, and
            // excluding alpha makes the whole interface opaque.
            color_write_mask: vk::ColorComponentFlags::R
                | vk::ColorComponentFlags::G
                | vk::ColorComponentFlags::B
                | vk::ColorComponentFlags::A,
            ..Default::default()
        };
        let color_blend = vk::PipelineColorBlendStateCreateInfo {
            attachment_count: desc.color_attachment_count,
            p_attachments: &blend_attachment,
            ..Default::default()
        };
        let rendering = vk::PipelineRenderingCreateInfo {
            color_attachment_count: desc.color_attachment_count,
            p_color_attachment_formats: target_formats.as_ptr(),
            ..Default::default()
        };
        let create_info = vk::GraphicsPipelineCreateInfo {
            stage_count: stages.len() as u32,
            p_next: &rendering as *const _ as *const _,
            p_stages: stages.as_ptr(),
            p_vertex_input_state: &vertex_input,
            p_input_assembly_state: &input_assembly,
            p_viewport_state: &viewport_state,
            p_rasterization_state: &rasterization,
            p_multisample_state: &multisample,
            p_depth_stencil_state: &depth_stencil,
            p_color_blend_state: &color_blend,
            p_dynamic_state: &dynamic,
            layout,
            render_pass: vk::RenderPass::null(),
            ..Default::default()
        };

        match device.create_graphics_pipelines(vk::PipelineCache::null(), &[create_info], None) {
            Ok(mut pipelines) => Ok(UiPipeline {
                pipeline: pipelines.remove(0),
                layout,
                descriptor_layout,
                vertex_module,
                fragment_module,
                cache_key,
            }),
            Err((_, e)) => {
                device.destroy_pipeline_layout(layout, None);
                device.destroy_descriptor_set_layout(descriptor_layout, None);
                device.destroy_shader_module(vertex_module, None);
                device.destroy_shader_module(fragment_module, None);
                Err(RenderError::Vk(e))
            }
        }
    }
}

/// The scissor for a clip rectangle given as four floats.
///
/// The rectangle is intersected with the surface and rounded outwards, because
/// a scissor that is one pixel too small clips a glyph's antialiased edge and
/// makes text look ragged.
pub fn scissor_from_bounds(
    min_x: f32,
    min_y: f32,
    max_x: f32,
    max_y: f32,
    width: u32,
    height: u32,
) -> vk::Rect2D {
    let sanitize = |v: f32| if v.is_finite() { v } else { 0.0 };
    let (min_x, min_y, max_x, max_y) = (
        sanitize(min_x).max(0.0),
        sanitize(min_y).max(0.0),
        sanitize(max_x).max(0.0),
        sanitize(max_y).max(0.0),
    );
    let left = min_x.floor().max(0.0).min(width as f32) as u32;
    let top = min_y.floor().max(0.0).min(height as f32) as u32;
    let right = max_x.ceil().max(0.0).min(width as f32) as u32;
    let bottom = max_y.ceil().max(0.0).min(height as f32) as u32;
    vk::Rect2D {
        offset: vk::Offset2D {
            x: left as i32,
            y: top as i32,
        },
        extent: vk::Extent2D {
            width: right.saturating_sub(left),
            height: bottom.saturating_sub(top),
        },
    }
}

/// A descriptor set holding one uniform buffer, which is all the UI pipeline
/// needs.
///
/// The existing `BindGroup` is built around a layout with a camera, a texture
/// and a sampler, so it cannot express "one uniform and nothing else" without
/// binding a texture the UI shader does not sample. A small dedicated type keeps
/// the pool sized to what the pipeline actually declares.
pub struct UiBindGroup {
    /// The allocated set.
    pub set: vk::DescriptorSet,
    /// The pool it was allocated from, which the caller must destroy.
    pub pool: vk::DescriptorPool,
}

impl std::fmt::Debug for UiBindGroup {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("UiBindGroup")
            .field("set", &self.set)
            .finish()
    }
}

impl UiBindGroup {
    /// Allocate a set from `layout` and point binding 0 at `buffer`.
    ///
    /// # Safety
    ///
    /// `device` must be live, `layout` must be the UI pipeline's descriptor set
    /// layout, and `buffer` must outlive the set.
    pub unsafe fn create(
        device: &ash::Device,
        layout: vk::DescriptorSetLayout,
        buffer: vk::Buffer,
        range: vk::DeviceSize,
    ) -> Result<UiBindGroup, RenderError> {
        let pool_sizes = [vk::DescriptorPoolSize {
            ty: vk::DescriptorType::UNIFORM_BUFFER,
            descriptor_count: 1,
        }];
        let pool_info = vk::DescriptorPoolCreateInfo {
            max_sets: 1,
            pool_size_count: pool_sizes.len() as u32,
            p_pool_sizes: pool_sizes.as_ptr(),
            ..Default::default()
        };
        let pool = unsafe { device.create_descriptor_pool(&pool_info, None) }?;

        let set_info = vk::DescriptorSetAllocateInfo {
            descriptor_pool: pool,
            p_set_layouts: &layout,
            descriptor_set_count: 1,
            ..Default::default()
        };
        let sets = match unsafe { device.allocate_descriptor_sets(&set_info) } {
            Ok(s) => s,
            Err(e) => {
                unsafe { device.destroy_descriptor_pool(pool, None) };
                return Err(RenderError::Vk(e));
            }
        };
        let info = [vk::DescriptorBufferInfo {
            buffer,
            offset: 0,
            range,
        }];
        let write = vk::WriteDescriptorSet {
            dst_set: sets[0],
            dst_binding: TRANSFORM_BINDING,
            descriptor_count: 1,
            p_buffer_info: info.as_ptr(),
            descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
            ..Default::default()
        };
        unsafe { device.update_descriptor_sets(&[write], &[]) };
        Ok(UiBindGroup { set: sets[0], pool })
    }

    /// Destroy the pool, which frees the set with it.
    ///
    /// # Safety
    ///
    /// The device must be alive and the set must be out of use.
    pub unsafe fn destroy(&self, device: &ash::Device) {
        unsafe { device.destroy_descriptor_pool(self.pool, None) };
    }
}

/// How many vertices the UI draws per frame before it is truncated.
///
/// A full 4K frame of dense UI is a few hundred thousand triangles; a million
/// vertices is well past that and bounds the damage a pathological frame can do
/// to the upload.
///
/// A multiple of three, because the pipeline is a triangle list and a vertex
/// count that is not leaves a partial triangle at the end — which is a
/// validation error, not a dropped fragment.
pub const MAX_UI_VERTICES: usize = 1_048_575;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_ui_vertex_is_12_bytes() {
        assert_eq!(std::mem::size_of::<UiVertex>(), 12);
        assert_eq!(UI_VERTEX_SIZE, 12);
    }

    #[test]
    fn the_vertex_casts_to_bytes() {
        let v = UiVertex {
            position: [1.0, 2.0],
            color: [3, 4, 5, 6],
        };
        assert_eq!(bytemuck::bytes_of(&v).len(), UI_VERTEX_SIZE as usize);
    }

    #[test]
    fn the_layout_places_both_attributes_inside_the_stride() {
        let l = ui_vertex_layout();
        assert_eq!(l.stride, UI_VERTEX_SIZE);
        assert_eq!(l.locations(), vec![0, 1]);
        assert!(
            l.used_bytes() <= l.stride,
            "an attribute must not reach past the stride"
        );
    }

    #[test]
    fn the_layout_is_valid() {
        assert!(ui_vertex_layout().validate().is_ok());
    }

    #[test]
    fn the_layout_has_no_padding_hole() {
        // The fields add up to the stride exactly, so there is no tail padding
        // for a bytemuck cast to expose.
        assert!(!ui_vertex_layout().has_padding());
    }

    #[test]
    fn the_attributes_do_not_overlap() {
        let l = ui_vertex_layout();
        let a = l.attributes;
        assert_eq!(a[0].offset, 0);
        assert_eq!(a[0].format.size(), 8);
        assert_eq!(a[1].offset, 8, "the colour follows the position");
        assert!(a[0].offset + a[0].format.size() <= a[1].offset);
    }

    #[test]
    fn the_transform_maps_pixels_onto_clip_space() {
        let t = ScreenTransform::for_size(800, 600);
        let x = 0.0 * t.scale_offset[0] + t.scale_offset[2];
        let y = 0.0 * t.scale_offset[1] + t.scale_offset[3];
        assert!(
            (x + 1.0).abs() < 1e-6,
            "the top-left pixel is clip x -1: {x}"
        );
        assert!(
            (y + 1.0).abs() < 1e-6,
            "the top-left pixel is clip y -1: {y}"
        );
    }

    #[test]
    fn the_transform_maps_the_far_corner_onto_plus_one() {
        let t = ScreenTransform::for_size(800, 600);
        let x = 800.0 * t.scale_offset[0] + t.scale_offset[2];
        let y = 600.0 * t.scale_offset[1] + t.scale_offset[3];
        assert!((x - 1.0).abs() < 1e-6, "{x}");
        assert!((y - 1.0).abs() < 1e-6, "{y}");
    }

    #[test]
    fn the_transform_maps_the_centre_onto_the_origin() {
        let t = ScreenTransform::for_size(800, 600);
        let x = 400.0 * t.scale_offset[0] + t.scale_offset[2];
        let y = 300.0 * t.scale_offset[1] + t.scale_offset[3];
        assert!(x.abs() < 1e-6, "{x}");
        assert!(y.abs() < 1e-6, "{y}");
    }

    #[test]
    fn a_zero_sized_surface_gives_a_finite_transform() {
        // A minimised window reports a zero extent. Dividing by it would give
        // infinity and every UI vertex would become NaN.
        let t = ScreenTransform::for_size(0, 0);
        assert!(t.scale_offset.iter().all(|v| v.is_finite()), "{t:?}");
    }

    #[test]
    fn a_zero_width_gives_a_zero_scale() {
        // Zero width cannot be mapped onto clip space, so the scale is zero and
        // the frame draws nothing. A NaN would be worse: it would blank the UI
        // and keep it blanked.
        let t = ScreenTransform::for_size(0, 600);
        assert_eq!(t.scale_offset[0], 0.0);
        assert!(t.scale_offset.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn a_zero_height_gives_a_zero_scale() {
        let t = ScreenTransform::for_size(800, 0);
        assert_eq!(t.scale_offset[1], 0.0);
        assert!(t.scale_offset.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn the_transform_is_sixteen_bytes() {
        assert_eq!(std::mem::size_of::<ScreenTransform>(), 16);
    }

    #[test]
    fn the_ui_description_fills_both_stages() {
        let spirv = vibe_shader::compile_entry_point(
            UI_SHADER,
            vibe_shader::CompileOptions::release(),
            Some((VERTEX_ENTRY, naga::ShaderStage::Vertex)),
        )
        .expect("the UI shader compiles");
        let desc = PipelineDesc {
            vertex_spirv: Some(spirv.clone()),
            fragment_spirv: Some(spirv),
            ..ui_pipeline_desc()
        };
        assert!(desc.validate().is_ok(), "{:?}", desc.validate().err());
    }

    #[test]
    fn the_pipeline_declares_no_push_constants() {
        // naga 0.30 cannot parse a push constant in WGSL, so a non-zero range
        // would be a range the shader can never read.
        assert_eq!(ui_pipeline_desc().push_constant_bytes, 0);
    }

    #[test]
    fn the_pipeline_uses_a_triangle_list() {
        assert_eq!(ui_pipeline_desc().primitive.topology, Topology::Triangles);
    }

    #[test]
    fn the_pipeline_alpha_blends() {
        assert_eq!(ui_pipeline_desc().blend, BlendState::AlphaBlend);
    }

    #[test]
    fn the_vertex_limit_is_a_multiple_of_three() {
        // A triangle list needs whole triangles; a limit that is not a multiple
        // of three means the last triangle is drawn from incomplete vertices.
        assert_eq!(MAX_UI_VERTICES % 3, 0);
    }

    #[test]
    fn the_vertex_limit_is_large_enough_for_a_full_screen_ui() {
        assert!(MAX_UI_VERTICES >= 100_000, "{MAX_UI_VERTICES}");
    }

    #[test]
    fn the_shader_declares_both_entry_points() {
        assert!(UI_SHADER.contains("fn vs_main"));
        assert!(UI_SHADER.contains("fn fs_main"));
    }

    #[test]
    fn the_shader_reads_the_transform_from_a_uniform() {
        // naga 0.30 cannot parse a push constant in WGSL, so the transform is a
        // uniform buffer at binding 0 and the pipeline declares exactly that one
        // set. The alternative — a declared set the shader never reads — makes
        // every draw invalid, because Vulkan still requires it to be bound.
        assert!(UI_SHADER.contains("var<uniform> transform"));
        assert!(UI_SHADER.contains("@group(0) @binding(0)"));
    }

    #[test]
    fn the_shader_declares_exactly_one_descriptor() {
        assert_eq!(
            UI_SHADER.matches("@binding(").count(),
            1,
            "one binding, and the pipeline layout declares one set"
        );
    }

    #[test]
    fn the_shader_samples_no_texture() {
        // The fragment stage returns the vertex colour directly. Sampling an
        // unbound font atlas here would fault on drivers that do not tolerate
        // it, and the glyphs are already rasterised into the vertices by egui.
        assert!(
            !UI_SHADER.contains("textureSample"),
            "the fragment stage must not sample"
        );
    }
}
