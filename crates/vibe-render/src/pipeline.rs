//! The 2D quad pipeline: one WGSL module and the Vulkan pipeline built from it.

use ash::vk;
use vibe_pipeline::{BlendState, PipelineDesc, vertex_layout_for_quad};
use vibe_shader::ShaderCompiler;

use crate::error::RenderError;

/// The quad shader, in WGSL.
///
/// The camera is a uniform buffer rather than push constants: the descriptor
/// set already exists for the texture and sampler, so the camera rides along in
/// it for no extra bind. The vertex stage reads it, the fragment stage samples
/// the texture.
pub const QUAD_SHADER: &str = r#"
struct Camera {
    view_projection: mat4x4<f32>,
}

struct QuadVertex {
    @location(0) position: vec2<f32>,
    @location(1) color: vec4<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) texture_index: u32,
}

struct VertexOut {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) @interpolate(flat) texture_index: u32,
}

@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var textures: texture_2d<f32>;
@group(0) @binding(2) var tex_sampler: sampler;

@vertex
fn vs_main(vertex: QuadVertex) -> VertexOut {
    var out: VertexOut;
    out.clip_position = camera.view_projection * vec4<f32>(vertex.position, 0.0, 1.0);
    out.color = vertex.color;
    out.uv = vertex.uv;
    out.texture_index = vertex.texture_index;
    return out;
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    return vec4<f32>(0.0, 1.0, 0.0, 1.0);
}
"#;

/// The name of the vertex entry point.
pub const VERTEX_ENTRY: &str = "vs_main";
/// The name of the fragment entry point.
pub const FRAGMENT_ENTRY: &str = "fs_main";

/// The descriptor bindings the quad shader expects.
pub const CAMERA_BINDING: u32 = 0;
pub const TEXTURE_BINDING: u32 = 1;
pub const SAMPLER_BINDING: u32 = 2;

/// A built quad pipeline and the objects it owns.
pub struct QuadPipeline {
    /// The pipeline itself.
    pub pipeline: vk::Pipeline,
    /// The layout the pipeline was built against.
    pub layout: vk::PipelineLayout,
    /// The descriptor set layout for the camera, texture array and sampler.
    pub descriptor_layout: vk::DescriptorSetLayout,
    /// The compiled vertex shader.
    pub vertex_module: vk::ShaderModule,
    /// The compiled fragment shader.
    pub fragment_module: vk::ShaderModule,
    /// The cache key of the description this was built from.
    pub cache_key: u64,
}

impl std::fmt::Debug for QuadPipeline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QuadPipeline")
            .field("pipeline", &self.pipeline)
            .field("cache_key", &self.cache_key)
            .finish()
    }
}

/// The description the 2D quad pipeline is built from.
pub fn quad_pipeline_desc() -> PipelineDesc {
    PipelineDesc {
        name: "quad2d".to_string(),
        layout: vertex_layout_for_quad(),
        // No push constants: the camera is a uniform buffer.
        push_constant_bytes: 0,
        // The batcher submits sprites back to front, so blending is what makes
        // the order meaningful and depth is off entirely.
        blend: BlendState::AlphaBlend,
        ..PipelineDesc::quad_2d("quad2d")
    }
}

/// A compiler that selects one entry point per stage.
///
/// The `ShaderCompiler` trait is source-in, SPIR-V-out, with no notion of which
/// entry point; a pipeline needs one module per stage, so this wraps the
/// injected compiler and narrows the output to a named entry.
struct EntryPointCompiler<'a, C> {
    inner: &'a C,
    entry: &'a str,
    stage: naga::ShaderStage,
}

impl<C: ShaderCompiler> ShaderCompiler for EntryPointCompiler<'_, C> {
    fn compile(
        &self,
        source: &str,
        options: vibe_shader::CompileOptions,
    ) -> Result<Vec<u32>, vibe_shader::ShaderError> {
        // Ask the injected compiler for the whole module first: that proves it
        // can handle this source, and a source it rejects becomes a pipeline
        // error here rather than a driver failure much later.
        self.inner.compile(source, options.clone())?;
        vibe_shader::compile_entry_point(source, options, Some((self.entry, self.stage)))
    }
}

/// Compile and build the quad pipeline.
///
/// # Safety
///
/// `device` must be a live logical device.
pub unsafe fn build_quad_pipeline<C: ShaderCompiler>(
    device: &ash::Device,
    compiler: &C,
) -> Result<QuadPipeline, RenderError> {
    use vibe_shader::CompileOptions;

    let vertex_spirv = EntryPointCompiler {
        inner: compiler,
        entry: VERTEX_ENTRY,
        stage: naga::ShaderStage::Vertex,
    }
    .compile(QUAD_SHADER, CompileOptions::release())
    .map_err(|e| RenderError::Pipeline(e.to_string()))?;
    let fragment_spirv = EntryPointCompiler {
        inner: compiler,
        entry: FRAGMENT_ENTRY,
        stage: naga::ShaderStage::Fragment,
    }
    .compile(QUAD_SHADER, CompileOptions::release())
    .map_err(|e| RenderError::Pipeline(e.to_string()))?;

    let desc = PipelineDesc {
        vertex_spirv: Some(vertex_spirv),
        fragment_spirv: Some(fragment_spirv),
        ..quad_pipeline_desc()
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

        // One combined image sampler for the texture array, plus the camera
        // buffer. The sampler is immutable, so a descriptor set holds only the
        // camera and the texture array is written once at startup.
        let bindings = [
            vk::DescriptorSetLayoutBinding {
                binding: CAMERA_BINDING,
                descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
                descriptor_count: 1,
                stage_flags: vk::ShaderStageFlags::VERTEX,
                ..Default::default()
            },
            vk::DescriptorSetLayoutBinding {
                binding: TEXTURE_BINDING,
                descriptor_type: vk::DescriptorType::SAMPLED_IMAGE,
                descriptor_count: 1,
                stage_flags: vk::ShaderStageFlags::FRAGMENT,
                ..Default::default()
            },
            vk::DescriptorSetLayoutBinding {
                binding: SAMPLER_BINDING,
                descriptor_type: vk::DescriptorType::SAMPLER,
                descriptor_count: 1,
                stage_flags: vk::ShaderStageFlags::FRAGMENT,
                ..Default::default()
            },
        ];
        let layout_info = vk::DescriptorSetLayoutCreateInfo {
            binding_count: bindings.len() as u32,
            p_bindings: bindings.as_ptr(),
            ..Default::default()
        };
        let descriptor_layout = device.create_descriptor_set_layout(&layout_info, None)?;

        // A zero-size range is a validation error, so it is declared only when
        // the pipeline actually uses push constants.
        let push = vk::PushConstantRange {
            stage_flags: vk::ShaderStageFlags::VERTEX,
            offset: 0,
            size: desc.push_constant_bytes,
        };
        let (range_count, range_ptr) = if desc.push_constant_bytes == 0 {
            (0, std::ptr::null())
        } else {
            (1, &push as *const vk::PushConstantRange)
        };
        let pipeline_layout_info = vk::PipelineLayoutCreateInfo {
            set_layout_count: 1,
            p_set_layouts: &descriptor_layout,
            push_constant_range_count: range_count,
            p_push_constant_ranges: range_ptr,
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
        // The viewport and scissor are dynamic: the window can be resized, and
        // a static viewport must additionally carry a non-zero size at
        // creation. Declaring them dynamic means the renderer sets both per
        // frame, and a missing set is a command-buffer recording error rather
        // than a silent one-pixel raster.
        // Bound to a local: the create info holds a pointer into it.
        let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
        let dynamic = vk::PipelineDynamicStateCreateInfo {
            dynamic_state_count: dynamic_states.len() as u32,
            p_dynamic_states: dynamic_states.as_ptr(),
            ..Default::default()
        };
        let viewport_state = vk::PipelineViewportStateCreateInfo {
            viewport_count: 1,
            ..Default::default()
        };
        let rasterization = desc.raster.vk();
        let multisample = vk::PipelineMultisampleStateCreateInfo {
            rasterization_samples: desc.samples.raster_samples,
            ..Default::default()
        };
        let depth_stencil = desc.depth.vk();
        // An enabled blend with default (zero) factors computes
        // src*0 + dst*0, which erases the draw instead of compositing it, so the
        // factors have to be written explicitly.
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
            // Every channel: a sprite with an alpha must still write RGB.
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

        let create_info = vk::GraphicsPipelineCreateInfo {
            stage_count: stages.len() as u32,
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
            // Dynamic rendering: the render pass is supplied at record time, so
            // this is null and compatibility is None.
            render_pass: vk::RenderPass::null(),
            ..Default::default()
        };
        let pipelines =
            device.create_graphics_pipelines(vk::PipelineCache::null(), &[create_info], None);
        let pipeline = match pipelines {
            Ok(mut p) => p.remove(0),
            Err((_, e)) => {
                device.destroy_pipeline_layout(layout, None);
                device.destroy_descriptor_set_layout(descriptor_layout, None);
                device.destroy_shader_module(vertex_module, None);
                device.destroy_shader_module(fragment_module, None);
                return Err(RenderError::Vk(e));
            }
        };

        log::info!("quad pipeline built (cache key {cache_key:#x})");
        Ok(QuadPipeline {
            pipeline,
            layout,
            descriptor_layout,
            vertex_module,
            fragment_module,
            cache_key,
        })
    }
}

fn create_module(device: &ash::Device, words: &[u32]) -> Result<vk::ShaderModule, RenderError> {
    let info = vk::ShaderModuleCreateInfo {
        code_size: words.len() * 4,
        p_code: words.as_ptr(),
        ..Default::default()
    };
    unsafe { device.create_shader_module(&info, None) }.map_err(RenderError::Vk)
}

#[cfg(test)]
mod tests {
    use super::*;
    use vibe_shader::{CompileOptions, looks_like_spirv};

    #[test]
    fn the_shader_compiles_for_both_stages() {
        for (name, stage) in [
            (VERTEX_ENTRY, naga::ShaderStage::Vertex),
            (FRAGMENT_ENTRY, naga::ShaderStage::Fragment),
        ] {
            let words = vibe_shader::compile_entry_point(
                QUAD_SHADER,
                CompileOptions::release(),
                Some((name, stage)),
            )
            .unwrap_or_else(|e| panic!("{name}: {e}"));
            assert!(looks_like_spirv(&words), "{name} produced no spirv");
        }
    }

    #[test]
    fn the_vertex_layout_matches_the_shader_locations() {
        let layout = vertex_layout_for_quad();
        // QuadVertex in WGSL: position 0, color 1, uv 2, texture_index 3.
        assert_eq!(layout.locations(), vec![0, 1, 2, 3]);
    }

    #[test]
    fn the_layout_matches_the_cpu_side_vertex() {
        assert_eq!(
            vertex_layout_for_quad().stride as usize,
            std::mem::size_of::<vibe_rhi::QuadVertex>()
        );
    }

    #[test]
    fn the_description_validates() {
        let mut desc = quad_pipeline_desc();
        desc.vertex_spirv = Some(vec![1]);
        desc.fragment_spirv = Some(vec![2]);
        assert!(desc.validate().is_ok());
    }

    #[test]
    fn the_pipeline_blends_and_has_no_depth() {
        let d = quad_pipeline_desc();
        assert_eq!(d.blend, BlendState::AlphaBlend);
        assert!(
            !d.depth.enabled,
            "2D sprites are sorted by the batcher, not by depth"
        );
    }

    #[test]
    fn the_pipeline_uses_no_push_constants() {
        // The camera is a uniform buffer, so the layout declares no push
        // constant range: a zero-size range is a validation error.
        assert_eq!(quad_pipeline_desc().push_constant_bytes, 0);
    }

    #[test]
    fn the_descriptor_bindings_are_distinct() {
        let bindings = [CAMERA_BINDING, TEXTURE_BINDING, SAMPLER_BINDING];
        assert_eq!(bindings.len(), 3);
        for a in bindings {
            for b in bindings {
                if a != b {
                    assert_ne!(a, b);
                }
            }
        }
    }

    #[test]
    fn the_cache_key_is_stable() {
        let a = quad_pipeline_desc();
        let b = quad_pipeline_desc();
        assert_eq!(a.cache_key(), b.cache_key());
    }

    #[test]
    fn the_shader_declares_the_bindings_the_layout_expects() {
        for binding in [CAMERA_BINDING, TEXTURE_BINDING, SAMPLER_BINDING] {
            assert!(
                QUAD_SHADER.contains(&format!("@binding({binding})")),
                "the shader must declare binding {binding}"
            );
        }
    }

    #[test]
    fn the_shader_uses_all_four_vertex_locations() {
        for location in 0..4 {
            assert!(
                QUAD_SHADER.contains(&format!("@location({location})")),
                "the shader must read vertex location {location}"
            );
        }
    }

    #[test]
    fn the_sampler_is_not_named_sampler() {
        // `sampler` is a reserved word in WGSL, which is a confusing error
        // because the declaration looks correct.
        assert!(!QUAD_SHADER.contains("var sampler:"));
        assert!(QUAD_SHADER.contains("tex_sampler"));
    }
}
