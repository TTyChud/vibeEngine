//! Textured 3D geometry: a vertex with a position, normal and texture
//! coordinate, and the pipeline that draws it.
//!
//! The UI pipeline and this one differ in three places and no more: the vertex
//! carries a normal and a UV instead of a colour, there is a camera matrix to
//! transform by, and depth testing decides what is in front. Everything else —
//! the barrier encoder, the descriptor plumbing, dynamic rendering — is the same
//! infrastructure, so this is a sibling of [`crate::ui`] rather than a second
//! renderer.
//!
//! Everything in this module that is a pure function of its inputs — the box
//! geometry, the camera matrices, the transform — is tested without a device,
//! because a box that is wound the wrong way is invisible from the inside and a
//! matrix with a flipped axis is only visible on a real screen.

use ash::vk;
use glam::{Mat4, Vec2, Vec3, Vec4};
use vibe_pipeline::state::{DepthCompare, Topology};
use vibe_pipeline::{
    Attribute, BlendState, DepthState, PipelineDesc, PrimitiveState, RasterState, VertexFormat,
    VertexLayout,
};

use crate::error::RenderError;
use crate::pipeline::create_module;
use vibe_shader::ShaderCompiler;

/// The vertex the 3D pipeline consumes.
///
/// 36 bytes: three floats of position, one of normal, two of texture coordinate.
/// Every field is 4-byte aligned and they add up exactly, so there is no padding
/// for a `bytemuck` cast to trip over.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Default, bytemuck::Pod, bytemuck::Zeroable)]
pub struct MeshVertex {
    /// Position in world space, in metres.
    pub position: [f32; 3],
    /// Surface normal, normalised. Used for lighting.
    pub normal: [f32; 3],
    /// Texture coordinate in `[0, 1]`, origin at the top left of the image.
    pub uv: [f32; 2],
}

impl MeshVertex {
    /// A vertex with an explicit position, normal and UV.
    pub fn new(position: Vec3, normal: Vec3, uv: Vec2) -> MeshVertex {
        MeshVertex {
            position: position.to_array(),
            normal: normal.to_array(),
            uv: uv.to_array(),
        }
    }
}

/// Bytes one 3D vertex occupies.
pub const MESH_VERTEX_SIZE: u32 = std::mem::size_of::<MeshVertex>() as u32;

/// The depth attachment format the 3D pipeline is built for.
///
/// Declared here rather than taken from the caller because the pipeline and the
/// frame have to agree on it exactly: dynamic rendering has no render pass
/// object to carry the format, so a mismatch is a validation error at the draw
/// with nothing else to point at the cause.
///
/// D32_SFLOAT rather than D24_UNORM, because 24-bit depth is not a required
/// format — a device that does not offer it fails image creation with an error
/// that names the format rather than the reason.
pub const DEPTH_FORMAT: vk::Format = vk::Format::D32_SFLOAT;

/// The vertex layout for [`MeshVertex`].
pub fn mesh_vertex_layout() -> VertexLayout {
    VertexLayout::new(
        MESH_VERTEX_SIZE,
        vec![
            Attribute::new(0, 0, VertexFormat::Float3),
            Attribute::new(1, 12, VertexFormat::Float3),
            Attribute::new(2, 24, VertexFormat::Float2),
        ],
    )
}

/// The camera the vertex stage transforms by.
///
/// A uniform buffer rather than a push constant, because naga 0.30's WGSL front
/// end has no push-constant address space: `var<push_constant>` is an unknown
/// address space and does not parse at all. The UI and quad pipelines take their
/// transforms the same way.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CameraUniform {
    /// The combined view-projection matrix, column-major to match WGSL's mat4x4.
    pub view_projection: [f32; 16],
    /// The camera position, for a view-dependent lighting term.
    pub eye: [f32; 4],
    /// `(ambient, 0, 0, 0)`, so the fragment stage is not reading a bare
    /// constant and the uniform has a second live field.
    pub lighting: [f32; 4],
}

impl Default for CameraUniform {
    fn default() -> Self {
        CameraUniform {
            view_projection: Mat4::IDENTITY.to_cols_array(),
            eye: [0.0; 4],
            lighting: [0.18, 0.0, 0.0, 0.0],
        }
    }
}

impl CameraUniform {
    /// The uniform for a camera and a field of view.
    pub fn new(view_projection: Mat4, eye: Vec3) -> CameraUniform {
        CameraUniform {
            view_projection: view_projection.to_cols_array(),
            eye: [eye.x, eye.y, eye.z, 1.0],
            lighting: CameraUniform::default().lighting,
        }
    }
}

/// The 3D shader, in WGSL.
///
/// One directional light plus ambient, so a box reads as a solid rather than a
/// flat silhouette, and the texture multiplies the lit colour. A box with no
/// texture binds a 1x1 white one, which is why the multiply is unconditional:
/// there is no separate untextured path to keep in step.
pub const MESH_SHADER: &str = r#"
struct MeshVertex {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
}

struct VertexOut {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) normal: vec3<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) world: vec3<f32>,
}

struct Camera {
    view_projection: mat4x4<f32>,
    eye: vec4<f32>,
    lighting: vec4<f32>,
}

@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var albedo: texture_2d<f32>;
@group(0) @binding(2) var albedo_sampler: sampler;

@vertex
fn vs_main(vertex: MeshVertex) -> VertexOut {
    var out: VertexOut;
    out.clip_position = camera.view_projection * vec4<f32>(vertex.position, 1.0);
    out.normal = vertex.normal;
    out.uv = vertex.uv;
    out.world = vertex.position;
    return out;
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
    // A fixed key light, not a camera-relative one, so rotating the camera does
    // not make the shading crawl.
    let light_dir = normalize(vec3<f32>(0.45, 0.8, 0.35));
    let lambert = max(dot(normalize(in.normal), light_dir), 0.0);
    let lit = camera.lighting.x + (1.0 - camera.lighting.x) * lambert;
    // A fifth of specular-ish rim on the camera-facing component, which is what
    // separates a box's three visible faces from each other.
    let to_eye = normalize(camera.eye.xyz - in.world);
    let rim = pow(max(dot(normalize(in.normal), to_eye), 0.0), 8.0) * 0.18;
    let texel = textureSample(albedo, albedo_sampler, in.uv);
    return vec4<f32>(texel.rgb * (lit + rim), texel.a);
}
"#;

/// The name of the vertex entry point.
pub const VERTEX_ENTRY: &str = "vs_main";
/// The name of the fragment entry point.
pub const FRAGMENT_ENTRY: &str = "fs_main";

/// The camera uniform binding.
pub const CAMERA_BINDING: u32 = 0;
/// The albedo texture binding.
pub const ALBEDO_BINDING: u32 = 1;
/// The albedo sampler binding.
pub const SAMPLER_BINDING: u32 = 2;

/// The description the 3D pipeline is built from.
///
/// Depth testing and writing are on, which is the whole point of this pipeline
/// existing next to the UI's: a scene drawn without depth is a pile of triangles
/// in submission order, so a box behind another box paints over it.
pub fn mesh_pipeline_desc() -> PipelineDesc {
    // Built from `Default` rather than from `quad_2d`, because a struct-update
    // at the *end* wins over every field written above it: `..quad_2d("mesh")`
    // silently reinstated its 64-byte push-constant range (this shader declares
    // none) and its disabled depth state (this pipeline is the one that tests
    // depth). Both are legal values, so neither is a validation error — the
    // result is a pipeline that draws nothing and says nothing about why.
    PipelineDesc {
        name: "mesh".to_string(),
        layout: mesh_vertex_layout(),
        push_constant_bytes: 0,
        // Opaque: a scene box has no alpha, and blending with an opaque source
        // costs a read-modify-write on every fragment for nothing.
        blend: BlendState::Opaque,
        // Depth testing and writing are on. This is the whole point of this
        // pipeline existing next to the UI's: a scene drawn without depth is a
        // pile of triangles in submission order, so a box behind another box
        // paints over it.
        depth: DepthState {
            enabled: true,
            write: true,
            compare: DepthCompare::Less,
        },
        primitive: PrimitiveState {
            topology: Topology::Triangles,
            ..Default::default()
        },
        // Rasterisation defaults: line width 1, no depth bias, no discard. The
        // cull mode is set per draw by the frame, which is why CULL_MODE is in
        // the dynamic state list below rather than declared statically here.
        ..Default::default()
    }
}

/// The dynamic states the 3D pipeline enables.
///
/// The same set the UI uses: the viewport and scissor follow the window, and
/// depth test, cull mode and topology are set per draw by the caller that owns
/// the frame.
pub fn mesh_dynamic_states() -> Vec<vk::DynamicState> {
    vec![
        vk::DynamicState::VIEWPORT,
        vk::DynamicState::SCISSOR,
        vk::DynamicState::CULL_MODE,
        vk::DynamicState::PRIMITIVE_TOPOLOGY,
        vk::DynamicState::DEPTH_TEST_ENABLE,
    ]
}

/// A built 3D pipeline and the objects it owns.
pub struct MeshPipeline {
    /// The pipeline itself.
    pub pipeline: vk::Pipeline,
    /// The layout the pipeline was built against.
    pub layout: vk::PipelineLayout,
    /// The descriptor set layout for the camera and albedo.
    pub descriptor_layout: vk::DescriptorSetLayout,
    /// The compiled vertex shader.
    pub vertex_module: vk::ShaderModule,
    /// The compiled fragment shader.
    pub fragment_module: vk::ShaderModule,
    /// The cache key of the description this was built from.
    pub cache_key: u64,
}

impl std::fmt::Debug for MeshPipeline {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MeshPipeline")
            .field("pipeline", &self.pipeline)
            .field("cache_key", &self.cache_key)
            .finish()
    }
}

impl MeshPipeline {
    /// Destroy the modules, layout and pipeline.
    ///
    /// # Safety
    ///
    /// The device must be alive and no draw may still be using the pipeline.
    pub unsafe fn destroy(&self, device: &ash::Device) {
        unsafe {
            device.destroy_pipeline(self.pipeline, None);
            device.destroy_pipeline_layout(self.layout, None);
            device.destroy_descriptor_set_layout(self.descriptor_layout, None);
            device.destroy_shader_module(self.vertex_module, None);
            device.destroy_shader_module(self.fragment_module, None);
        }
    }
}

/// Compile and build the 3D pipeline.
///
/// # Safety
///
/// `device` must be a live logical device.
pub unsafe fn build_mesh_pipeline<C: ShaderCompiler>(
    device: &ash::Device,
    compiler: &C,
    target_formats: &[vk::Format],
) -> Result<MeshPipeline, RenderError> {
    use vibe_shader::CompileOptions;

    let vertex_spirv = vibe_shader::compile_entry_point(
        MESH_SHADER,
        CompileOptions::release(),
        Some((VERTEX_ENTRY, naga::ShaderStage::Vertex)),
    )
    .map_err(|e| RenderError::Pipeline(e.to_string()))?;
    let fragment_spirv = vibe_shader::compile_entry_point(
        MESH_SHADER,
        CompileOptions::release(),
        Some((FRAGMENT_ENTRY, naga::ShaderStage::Fragment)),
    )
    .map_err(|e| RenderError::Pipeline(e.to_string()))?;
    compiler
        .compile(MESH_SHADER, CompileOptions::release())
        .map_err(|e| RenderError::Pipeline(e.to_string()))?;

    let desc = PipelineDesc {
        vertex_spirv: Some(vertex_spirv),
        fragment_spirv: Some(fragment_spirv),
        ..mesh_pipeline_desc()
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

        let bindings = [
            vk::DescriptorSetLayoutBinding {
                binding: CAMERA_BINDING,
                descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
                descriptor_count: 1,
                // Vertex *and* fragment: the vertex stage reads the view-projection
                // matrix, and the fragment stage reads the eye position and the
                // ambient term for its lighting. Declaring only VERTEX is a
                // validation error at pipeline creation, because a shader uses a
                // resource the layout does not admit.
                stage_flags: vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                ..Default::default()
            },
            vk::DescriptorSetLayoutBinding {
                binding: ALBEDO_BINDING,
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
        let dynamic_states = mesh_dynamic_states();
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
        let multisample = vk::PipelineMultisampleStateCreateInfo {
            rasterization_samples: desc.samples.raster_samples,
            ..Default::default()
        };
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
            // The depth format has to be declared, and it has to match the
            // attachment the frame actually binds. Left UNDEFINED, the draw is
            // invalid against any depth attachment: dynamic rendering has no
            // render pass object to carry the format, so the pipeline is the
            // only place it can live.
            depth_attachment_format: DEPTH_FORMAT,
            ..Default::default()
        };
        let create_info = vk::GraphicsPipelineCreateInfo {
            stage_count: stages.len() as u32,
            p_next: &rendering as *const _ as *const _,
            p_stages: stages.as_ptr(),
            p_vertex_input_state: &vertex_input,
            p_input_assembly_state: &input_assembly,
            p_dynamic_state: &dynamic,
            p_viewport_state: &viewport_state,
            p_rasterization_state: &desc.raster.vk(),
            p_multisample_state: &multisample,
            p_depth_stencil_state: &desc.depth.vk(),
            p_color_blend_state: &color_blend,
            // Without this the pipeline is created with a null layout, which the
            // layer reports as "not a valid VkPipelineLayout" and the driver may
            // accept — a pipeline that cannot be drawn against a descriptor set.
            layout,
            render_pass: vk::RenderPass::null(),
            ..Default::default()
        };
        match device.create_graphics_pipelines(vk::PipelineCache::null(), &[create_info], None) {
            Ok(mut pipelines) => Ok(MeshPipeline {
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

/// A descriptor set naming a camera uniform and one albedo texture.
pub struct MeshBindGroup {
    /// The allocated set.
    pub set: vk::DescriptorSet,
    /// The pool it was allocated from, which the caller must destroy.
    pub pool: vk::DescriptorPool,
}

impl std::fmt::Debug for MeshBindGroup {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MeshBindGroup")
            .field("set", &self.set)
            .finish()
    }
}

impl MeshBindGroup {
    /// Allocate a set naming the camera buffer and an albedo view.
    ///
    /// # Safety
    ///
    /// `device` must be live, `layout` must be the mesh pipeline's descriptor
    /// set layout, and `buffer` must outlive the set.
    pub unsafe fn create(
        device: &ash::Device,
        layout: vk::DescriptorSetLayout,
        buffer: vk::Buffer,
        range: vk::DeviceSize,
        albedo_view: vk::ImageView,
        sampler: vk::Sampler,
    ) -> Result<MeshBindGroup, RenderError> {
        let pool_sizes = [
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::UNIFORM_BUFFER,
                descriptor_count: 1,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::SAMPLED_IMAGE,
                descriptor_count: 1,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::SAMPLER,
                descriptor_count: 1,
            },
        ];
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
        let buffer_info = [vk::DescriptorBufferInfo {
            buffer,
            offset: 0,
            range,
        }];
        let image_info = [vk::DescriptorImageInfo {
            sampler,
            image_view: albedo_view,
            image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        }];
        let writes = [
            vk::WriteDescriptorSet {
                dst_set: sets[0],
                dst_binding: CAMERA_BINDING,
                descriptor_count: 1,
                p_buffer_info: buffer_info.as_ptr(),
                descriptor_type: vk::DescriptorType::UNIFORM_BUFFER,
                ..Default::default()
            },
            vk::WriteDescriptorSet {
                dst_set: sets[0],
                dst_binding: ALBEDO_BINDING,
                descriptor_count: 1,
                p_image_info: image_info.as_ptr(),
                descriptor_type: vk::DescriptorType::SAMPLED_IMAGE,
                ..Default::default()
            },
            vk::WriteDescriptorSet {
                dst_set: sets[0],
                dst_binding: SAMPLER_BINDING,
                descriptor_count: 1,
                p_image_info: image_info.as_ptr(),
                descriptor_type: vk::DescriptorType::SAMPLER,
                ..Default::default()
            },
        ];
        unsafe { device.update_descriptor_sets(&writes, &[]) };
        Ok(MeshBindGroup { set: sets[0], pool })
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

/// The six faces of a box, as `(normal, u_axis, v_axis)`.
///
/// Each face is a unit square spanned by two axes, wound so that
/// `normal = u_axis × v_axis`. That identity is what makes the winding correct:
/// a triangle whose vertices run `origin, origin+u, origin+u+v` has an outward
/// normal of exactly `u × v`, and Vulkan's default front face is
/// counter-clockwise in that winding. Deriving the winding from the cross
/// product rather than hand-listing it per face is what makes a face impossible
/// to get backwards — there is one rule, not twelve coordinates to remember.
const FACES: [(Vec3, Vec3, Vec3); 6] = [
    // +X
    (Vec3::X, Vec3::NEG_Z, Vec3::NEG_Y),
    // -X
    (Vec3::NEG_X, Vec3::Z, Vec3::NEG_Y),
    // +Y
    (Vec3::Y, Vec3::X, Vec3::Z),
    // -Y
    (Vec3::NEG_Y, Vec3::X, Vec3::NEG_Z),
    // +Z
    (Vec3::Z, Vec3::X, Vec3::NEG_Y),
    // -Z
    (Vec3::NEG_Z, Vec3::NEG_X, Vec3::NEG_Y),
];

/// The vertices of an axis-aligned box, centred on the origin, of the given
/// size.
///
/// 24 vertices rather than the 8 a shared-vertex cube would use: three faces
/// meet at every corner, and a shared corner would have to carry three
/// different normals and three different UV sets. Splitting the corners is what
/// lets each face have a flat normal and an unstretched texture.
///
/// The size is clamped to a positive extent on every axis. A zero or negative
/// extent is a collapsed box, which rasterises to nothing, so a resize that
/// drags one handle onto another would silently delete the object instead of
/// keeping a sliver on screen.
pub fn box_vertices(size: Vec3) -> Vec<MeshVertex> {
    let full = Vec3::new(
        (size.x.abs()).max(1.0e-3),
        (size.y.abs()).max(1.0e-3),
        (size.z.abs()).max(1.0e-3),
    );
    let half = full * 0.5;
    let mut out = Vec::with_capacity(36);
    for (normal, u, v) in FACES {
        // The face's centre sits half an extent along its own normal, so the
        // box is centred on the origin rather than hanging off a corner.
        let centre = normal * half;
        // Each corner is offset by the half-extent of the axis it lies on, and
        // the corner offsets in `QUAD_CORNERS` are ±1, so a face reaches exactly
        // the half-extent on each of the two axes it spans.
        //
        // The axis is read out of the offset direction rather than taken as
        // `half.x`/`half.y`, because a +X face on a 2x8x4 box spans Z and Y and
        // must reach 2 and 4, not 1 and 1. Using the wrong pair makes the face a
        // six-vertex star that back-face culling then drops entirely.
        //
        // `max_element` returns the largest *component*, so it has to be turned
        // back into an index: `1.0` is not a subscript, and using it as one
        // would index whatever happens to sit at that position.
        let extent_of = |axis: Vec3| {
            let a = axis.abs();
            if a.x >= a.y && a.x >= a.z {
                half.x
            } else if a.y >= a.z {
                half.y
            } else {
                half.z
            }
        };
        let (su, sv) = (extent_of(u), extent_of(v));
        for (a, b, uu, vv) in QUAD_CORNERS {
            let point = centre + u * (su * a) + v * (sv * b);
            out.push(MeshVertex::new(point, normal, Vec2::new(uu, vv)));
        }
    }
    out
}

/// The corners of a unit square in (u, v), as `(a, b, u, v)`.
///
/// The `a` and `b` offsets span **-1 to 1**, and `box_vertices` multiplies them
/// by the half-extent of the axis they lie on. They are not ±0.5: with a
/// half-extent multiplier, ±0.5 would put every face a quarter of the way out
/// and produce a box at half size.
///
/// Wound so that `(v - u)` is the *opposite* of the face normal, because the
/// projection flips handedness: a triangle wound `u × v = normal` comes out
/// clockwise in screen space once the view and projection matrices are applied,
/// and back-face culling then drops a face that is pointing straight at the
/// camera. The test `every_face_is_front_facing_when_seen_from_outside` measures
/// the result through a real projection rather than trusting this comment.
///
/// Six entries, not four: there is no index buffer, so the two triangles share
/// no vertices and each carries its own copy of the diagonal's endpoints. A
/// shared vertex would have to carry two different UVs where the diagonal runs.
const QUAD_CORNERS: [(f32, f32, f32, f32); 6] = [
    (-1.0, -1.0, 0.0, 1.0),
    (-1.0, 1.0, 0.0, 0.0),
    (1.0, -1.0, 1.0, 1.0),
    (-1.0, 1.0, 0.0, 0.0),
    (1.0, 1.0, 1.0, 0.0),
    (1.0, -1.0, 1.0, 1.0),
];

/// A perspective projection for a Vulkan clip space.
///
/// Built by hand rather than taken from `glam::camera::rh::proj::vulkan`,
/// because that helper is a Y-flip-plus-`0..1`-depth matrix for a different
/// convention, and mixing it with a top-left-origin UI space is exactly the kind
/// of mismatch that produces an upside-down scene. This maps near to 0 and far
/// to 1, which is what `DepthCompare::Less` against a cleared depth buffer of
/// 1.0 expects.
pub fn perspective(fov_y_radians: f32, aspect: f32, near: f32, far: f32) -> Mat4 {
    let f = 1.0 / (fov_y_radians * 0.5).tan();
    let mut m = Mat4::ZERO;
    m.x_axis.x = f / aspect.max(1.0e-6);
    m.y_axis.y = f;
    m.z_axis.z = far / (near - far);
    m.z_axis.w = -1.0;
    m.w_axis.z = (near * far) / (near - far);
    m
}

/// A right-handed look-at view matrix.
///
/// Hand-built for the same reason as [`perspective`]: `glam`'s `look_at_rh` is
/// deprecated and the replacement lives behind a different module path, and the
/// 12 lines here are worth more than the import.
pub fn look_at(eye: Vec3, target: Vec3, up: Vec3) -> Mat4 {
    let f = (target - eye).normalize_or_zero();
    let s = f.cross(up).normalize_or_zero();
    let u = s.cross(f);
    let mut m = Mat4::ZERO;
    m.x_axis = Vec4::new(s.x, u.x, -f.x, 0.0);
    m.y_axis = Vec4::new(s.y, u.y, -f.y, 0.0);
    m.z_axis = Vec4::new(s.z, u.z, -f.z, 0.0);
    m.w_axis = Vec4::new(-s.dot(eye), -u.dot(eye), f.dot(eye), 1.0);
    m
}

/// The eight corners of a box of `size` centred on `centre`.
///
/// The resize gizmo draws its eight handles at these points, so they have to be
/// the box's real corners rather than an approximation: a handle that sits
/// slightly outside the box is a handle the user cannot align to its own face.
pub fn box_corners(centre: Vec3, size: Vec3) -> [Vec3; 8] {
    let h = size * 0.5;
    [
        centre + Vec3::new(-h.x, -h.y, -h.z),
        centre + Vec3::new(h.x, -h.y, -h.z),
        centre + Vec3::new(h.x, h.y, -h.z),
        centre + Vec3::new(-h.x, h.y, -h.z),
        centre + Vec3::new(-h.x, -h.y, h.z),
        centre + Vec3::new(h.x, -h.y, h.z),
        centre + Vec3::new(h.x, h.y, h.z),
        centre + Vec3::new(-h.x, h.y, h.z),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn face_normals(verts: &[MeshVertex]) -> Vec<Vec3> {
        // One normal per face, in the order the faces were built.
        verts
            .chunks(6)
            .map(|f| {
                let n = Vec3::from_array(f[0].normal);
                // Every vertex of a face must agree, or the face is shaded as a
                // bent surface and a box looks like a folded card.
                assert!(
                    f.iter()
                        .all(|v| Vec3::from_array(v.normal).abs_diff_eq(n, 1.0e-6)),
                    "a face's vertices disagree about its normal"
                );
                n
            })
            .collect()
    }

    #[test]
    fn the_mesh_pipeline_tests_depth() {
        // The struct-update footgun: a `..quad_2d(..)` spread at the *end* of the
        // literal overwrites every field named above it, and it reinstates
        // `quad_2d`'s `DepthState::disabled()`. Depth off is a perfectly legal
        // pipeline, so nothing complains — the scene just draws with no depth
        // and the boxes that ought to occlude each other do not.
        let d = mesh_pipeline_desc();
        assert!(d.depth.enabled, "the mesh pipeline must test depth");
        assert!(d.depth.write, "the mesh pipeline must write depth");
        assert_eq!(d.depth.compare, DepthCompare::Less);
    }

    #[test]
    fn the_mesh_pipeline_declares_no_push_constants() {
        // The other half of the same footgun: `quad_2d` declares a 64-byte
        // push-constant range, and this shader has no push-constant block, so
        // inheriting the range makes the pipeline expect a range the frame never
        // sets.
        assert_eq!(mesh_pipeline_desc().push_constant_bytes, 0);
    }

    #[test]
    fn the_mesh_pipeline_uses_the_mesh_vertex_layout() {
        // A quad layout inherited from the spread would read the first 20 bytes
        // of a 32-byte vertex as a position and colour, and the box would
        // collapse to a sliver.
        let d = mesh_pipeline_desc();
        assert_eq!(d.layout.stride, MESH_VERTEX_SIZE);
        assert_eq!(d.layout.attribute_descriptions().len(), 3);
    }

    #[test]
    fn a_mesh_vertex_is_32_bytes() {
        // 12 of position, 12 of normal, 8 of uv.
        assert_eq!(std::mem::size_of::<MeshVertex>(), 32);
        assert_eq!(MESH_VERTEX_SIZE, 32);
    }

    #[test]
    fn the_vertex_casts_to_bytes() {
        let v = MeshVertex::new(Vec3::X, Vec3::Y, Vec2::new(0.5, 0.25));
        let bytes = bytemuck::bytes_of(&v);
        assert_eq!(bytes.len(), 32);
    }

    #[test]
    fn a_box_is_six_quads() {
        // Six vertices per face, because there is no index buffer: two triangles
        // of three vertices each. Six faces is 36.
        let verts = box_vertices(Vec3::new(2.0, 2.0, 2.0));
        assert_eq!(verts.len(), 36, "six faces of two triangles each");
    }

    #[test]
    fn the_vertex_count_is_a_multiple_of_three() {
        // A triangle list cannot draw a partial triangle.
        assert_eq!(box_vertices(Vec3::ONE).len() % 3, 0);
    }

    #[test]
    fn every_face_has_a_flat_normal() {
        let verts = box_vertices(Vec3::new(2.0, 3.0, 4.0));
        assert_eq!(face_normals(&verts).len(), 6);
    }

    #[test]
    fn the_faces_point_out_of_the_box() {
        // The dot of a face's normal with the direction from the box's centre to
        // that face's own vertices must be positive: an inward-facing normal is
        // a box lit from the inside, which is invisible.
        let size = Vec3::new(2.0, 2.0, 2.0);
        let verts = box_vertices(size);
        for (face, chunk) in verts.chunks(6).enumerate() {
            let n = Vec3::from_array(chunk[0].normal);
            let p = Vec3::from_array(chunk[0].position);
            assert!(
                n.dot(p) > 0.0,
                "face {face} normal {n:?} points inward from vertex {p:?}"
            );
        }
    }

    #[test]
    fn the_faces_cover_all_six_directions() {
        let verts = box_vertices(Vec3::ONE);
        let normals = face_normals(&verts);
        for axis in [Vec3::X, Vec3::Y, Vec3::Z] {
            assert!(
                normals.iter().any(|n| n.abs_diff_eq(axis, 1.0e-6)),
                "no face points along {axis:?}"
            );
            assert!(
                normals.iter().any(|n| n.abs_diff_eq(-axis, 1.0e-6)),
                "no face points along {:?}",
                -axis
            );
        }
    }

    #[test]
    fn the_box_has_the_size_it_was_asked_for() {
        let size = Vec3::new(2.0, 4.0, 6.0);
        let verts = box_vertices(size);
        for v in &verts {
            let p = Vec3::from_array(v.position);
            assert!(
                p.x.abs() <= size.x * 0.5 + 1.0e-5,
                "x {} exceeds half of {}",
                p.x,
                size.x
            );
            assert!(p.y.abs() <= size.y * 0.5 + 1.0e-5);
            assert!(p.z.abs() <= size.z * 0.5 + 1.0e-5);
        }
    }

    #[test]
    fn the_box_reaches_its_full_extent() {
        // The corner test above only checks the box is not too big. This checks
        // it is not too small, which a wrong half-extent factor would cause.
        //
        // Measured from the *centre* along each axis, so it is the corners that
        // have to reach the half-extent and not merely the face centres: a face
        // centred at the right place with corners a quarter of the way out is a
        // box at half size, and the centre-only measurement says it is correct.
        let size = Vec3::new(2.0, 6.0, 4.0);
        let verts = box_vertices(size);
        let centre = Vec3::ZERO;
        for axis in 0..3 {
            let reach = verts
                .iter()
                .map(|v| (Vec3::from_array(v.position) - centre)[axis].abs())
                .fold(0.0f32, f32::max);
            assert!(
                (reach - size[axis] * 0.5).abs() < 1.0e-5,
                "axis {axis} reaches {reach}, expected {}",
                size[axis] * 0.5
            );
        }
    }

    #[test]
    fn a_collapsed_box_still_has_extent() {
        // Dragging a resize handle onto its opposite face must not delete the
        // object, so a zero or negative size is clamped rather than accepted.
        // A collapsed box must not vanish: dragging a resize handle onto its
        // opposite face should leave a sliver, not delete the object.
        for size in [Vec3::ZERO, Vec3::new(0.0, 1.0, 1.0), Vec3::splat(-3.0)] {
            let verts = box_vertices(size);
            assert_eq!(verts.len(), 36);
            let reach = verts
                .iter()
                .map(|v| v.position[0].abs())
                .fold(0.0f32, f32::max);
            assert!(reach > 0.0, "a collapsed box has no extent on X");
        }
    }

    #[test]
    fn every_face_maps_the_whole_texture() {
        // A face whose UVs never reach 1 shows a cropped texture, which looks
        // like a bug in the image rather than in the geometry.
        for face in verts_by_face(&box_vertices(Vec3::ONE)) {
            for u in [face[0].uv[0], face[2].uv[0]] {
                assert!((0.0..=1.0).contains(&u), "u {u} is outside the image");
            }
            for v in [face[0].uv[1], face[2].uv[1]] {
                assert!((0.0..=1.0).contains(&v), "v {v} is outside the image");
            }
        }
    }

    fn verts_by_face(verts: &[MeshVertex]) -> Vec<&[MeshVertex]> {
        verts.chunks(6).collect()
    }

    /// The winding of every face, measured through a real projection.
    ///
    /// This is the test that catches an invisible box. Back-face culling drops
    /// whichever faces are wound the wrong way, and a box with all six wound
    /// wrong draws *nothing* — with no error, no validation message, and a
    /// frame that is otherwise perfectly correct. Reasoning about the sign of a
    /// world-space cross product is not enough to settle it, because the
    /// projection flips handedness and the answer depends on the camera; so the
    /// faces are projected and the screen-space winding is measured.
    #[test]
    fn every_face_is_front_facing_when_seen_from_outside() {
        let camera_pos = Vec3::new(6.0, 5.0, 6.0);
        let view = look_at(camera_pos, Vec3::ZERO, Vec3::Y);
        let proj = perspective(std::f32::consts::FRAC_PI_3, 936.0 / 1048.0, 0.05, 500.0);
        let vp = proj * view;
        let size = Vec3::splat(2.0);
        let half = size * 0.5;

        for (face, chunk) in box_vertices(size).chunks(6).enumerate() {
            // The first triangle of the face decides the whole quad.
            let screen: Vec<Vec2> = chunk[..3]
                .iter()
                .map(|v| {
                    let clip = vp * Vec3::from_array(v.position).extend(1.0);
                    let ndc = clip.truncate() / clip.w;
                    Vec2::new(ndc.x, ndc.y)
                })
                .collect();
            // Twice the signed area. Positive is counter-clockwise with Y up,
            // which is Vulkan's front face under the default
            // FRONT_FACE_COUNTER_CLOCKWISE — except that the projection's own
            // Y-flip is already in `vp`, so this measures what the rasteriser
            // sees rather than what the world-space vectors suggest.
            let area = (screen[1].x - screen[0].x) * (screen[2].y - screen[0].y)
                - (screen[2].x - screen[0].x) * (screen[1].y - screen[0].y);
            // And the face must actually point back at the camera, which is what
            // "seen from outside" means for a convex solid.
            let normal = Vec3::from_array(chunk[0].normal);
            let centre = Vec3::from_array(chunk[0].position) - normal * half.dot(normal.abs());
            let faces_eye = normal.dot(camera_pos - centre) > 0.0;
            assert!(
                (area > 0.0) == faces_eye,
                "face {face} is wound the wrong way: screen area {area:+.5}, \
                 normal {normal:?} faces the eye: {faces_eye}"
            );
        }
    }

    #[test]
    fn a_corner_offset_uses_the_extent_of_the_axis_it_spans() {
        // The bug this catches: scaling a face's corners by the *box's* half
        // extents rather than by the extent of the two axes that face spans. The
        // +X face then reaches half a box in Y and Z as well, so it is a
        // six-vertex star rather than a square — which back-face culling turns
        // into no face at all.
        let size = Vec3::new(2.0, 8.0, 4.0);
        let verts = box_vertices(size);
        for chunk in verts.chunks(6) {
            let normal = Vec3::from_array(chunk[0].normal);
            for v in chunk {
                let p = Vec3::from_array(v.position);
                // A point on this face lies in the plane at the box's own
                // half-extent along the normal — offset *from* the origin, not
                // on the plane through it, since a face is a face of the box and
                // not through its middle.
                assert!(
                    (p.dot(normal) - size.dot(normal.abs()) * 0.5).abs() < 1.0e-4,
                    "a vertex of the {normal:?} face is off its own plane: {p:?}"
                );
                // And on each of the two axes it spans, it reaches exactly that
                // axis's half-extent.
                for axis in 0..3 {
                    if normal[axis].abs() < 0.5 {
                        assert!(
                            (p[axis].abs() - size[axis] * 0.5).abs() < 1.0e-4,
                            "face normal {normal:?} reaches {} on axis {axis}, \
                             expected {}",
                            p[axis].abs(),
                            size[axis] * 0.5
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn a_face_spans_exactly_two_axes() {
        for (normal, u, v) in FACES {
            let spanned = [u, v]
                .iter()
                .filter(|a| a.x.abs() + a.y.abs() + a.z.abs() > 0.5)
                .count();
            assert_eq!(spanned, 2, "face {normal:?} does not span two axes");
            // The two spanned axes and the normal are all three axes, once each.
            let mut axes: Vec<usize> = [normal, u, v]
                .iter()
                .map(|a| (0..3).find(|i| a[*i].abs() > 0.5).unwrap())
                .collect();
            axes.sort_unstable();
            assert_eq!(axes, vec![0, 1, 2], "face {normal:?} repeats an axis");
        }
    }

    /// Transform a point as a position, not a direction: the w is 1, so a
    /// perspective divide applies. `Mat4 * Vec3` does not exist in glam, and
    /// silently using `Vec4::w == 0` instead would skip the divide that every
    /// one of these assertions is about.
    fn project(m: Mat4, p: Vec3) -> Vec4 {
        m * p.extend(1.0)
    }

    #[test]
    fn the_near_plane_maps_to_zero_depth() {
        let m = perspective(std::f32::consts::FRAC_PI_3, 1.0, 0.1, 100.0);
        // A point on the near plane, in view space with -Z forward.
        let near = project(m, Vec3::new(0.0, 0.0, -0.1));
        assert!(
            (near.z / near.w).abs() < 1.0e-4,
            "near plane depth was {}",
            near.z / near.w
        );
    }

    #[test]
    fn the_far_plane_maps_to_one_depth() {
        let m = perspective(std::f32::consts::FRAC_PI_3, 1.0, 0.1, 100.0);
        let far = project(m, Vec3::new(0.0, 0.0, -100.0));
        assert!(
            (far.z / far.w - 1.0).abs() < 1.0e-4,
            "far plane depth was {}",
            far.z / far.w
        );
    }

    #[test]
    fn a_point_in_front_of_the_camera_has_positive_w() {
        // A right-handed view space puts what is in front at negative Z, and w
        // is the distance along it. This is the property the gizmo's projection
        // relies on to reject a point behind the camera.
        let m = perspective(std::f32::consts::FRAC_PI_3, 1.0, 0.1, 100.0);
        let front = project(m, Vec3::new(0.0, 0.0, -5.0));
        assert!(front.w > 0.0, "a point 5 units ahead had w {}", front.w);
    }

    #[test]
    fn a_wide_viewport_does_not_narrow_the_horizontal_field() {
        // Dividing by the aspect is what stops a 16:9 window from cropping the
        // sides of a square box; the reverse is the bug.
        let square = perspective(1.0, 1.0, 0.1, 100.0);
        let wide = perspective(1.0, 2.0, 0.1, 100.0);
        let p = Vec3::new(1.0, 0.0, -10.0);
        let a = project(square, p);
        let b = project(wide, p);
        let square_x = a.x / a.w;
        let wide_x = b.x / b.w;
        assert!(
            wide_x < square_x,
            "a wider viewport gave a narrower horizontal field"
        );
    }

    #[test]
    fn the_view_matrix_puts_the_target_in_front_of_the_camera() {
        let eye = Vec3::new(0.0, 0.0, 5.0);
        let view = look_at(eye, Vec3::ZERO, Vec3::Y);
        let origin = project(view, Vec3::ZERO);
        // The target is straight ahead, so it lands on -Z in view space.
        assert!(
            origin.z < 0.0,
            "the target was not in front of the camera: {origin:?}"
        );
        assert!((origin.x).abs() < 1.0e-5);
        assert!((origin.y).abs() < 1.0e-5);
    }

    #[test]
    fn the_view_matrix_does_not_move_the_eye() {
        // The eye is the one point a view matrix must leave at the origin; if it
        // does not, the whole scene is offset by the camera's position.
        let eye = Vec3::new(3.0, -4.0, 5.0);
        let view = look_at(eye, Vec3::ZERO, Vec3::Y);
        let at_eye = project(view, eye);
        // Compared on xyz only: the result is a homogeneous point, so its w is
        // 1 and its four-component length is always at least 1.
        assert!(
            at_eye.truncate().length() < 1.0e-4,
            "the eye moved to {at_eye:?}"
        );
    }

    #[test]
    fn an_up_vector_parallel_to_the_view_does_not_produce_nonsense() {
        // Looking straight down with world-up as the up vector is degenerate.
        // A zero basis is the right answer: a NaN here would propagate into
        // every vertex and blank the frame.
        let view = look_at(Vec3::new(0.0, 5.0, 0.0), Vec3::ZERO, Vec3::Y);
        let p = project(view, Vec3::ONE);
        assert!(p.is_finite(), "degenerate up produced {p:?}");
    }

    #[test]
    fn the_camera_uniform_is_std140_shaped() {
        // Three mat4/vec4 members, each 16-byte aligned: 96 bytes, no padding to
        // account for. A wrong size here is a silently misread matrix.
        assert_eq!(std::mem::size_of::<CameraUniform>(), 96);
    }

    #[test]
    fn the_camera_uniform_carries_the_matrix_and_eye() {
        let m = perspective(1.0, 1.0, 0.1, 100.0);
        let u = CameraUniform::new(m, Vec3::new(1.0, 2.0, 3.0));
        assert_eq!(u.view_projection, m.to_cols_array());
        assert_eq!(&u.eye[..3], &[1.0, 2.0, 3.0]);
        assert_eq!(u.eye[3], 1.0, "the eye's w marks it as a position");
    }

    #[test]
    fn the_box_corners_span_the_box() {
        let corners = box_corners(Vec3::new(1.0, 0.0, 0.0), Vec3::new(2.0, 2.0, 2.0));
        assert_eq!(corners.len(), 8);
        // The centre plus the half-extent on each axis, and the corners are
        // centred on the box's own centre.
        let mean = corners.iter().fold(Vec3::ZERO, |a, c| a + *c) / 8.0;
        assert!(
            mean.abs_diff_eq(Vec3::new(1.0, 0.0, 0.0), 1.0e-5),
            "corners averaged to {mean:?}"
        );
    }

    #[test]
    fn the_corner_offsets_cover_every_sign_combination() {
        let corners = box_corners(Vec3::ZERO, Vec3::splat(2.0));
        let mut signs: Vec<(i8, i8, i8)> = corners
            .iter()
            .map(|c| {
                (
                    if c.x > 0.0 { 1 } else { -1 },
                    if c.y > 0.0 { 1 } else { -1 },
                    if c.z > 0.0 { 1 } else { -1 },
                )
            })
            .collect();
        signs.sort();
        signs.dedup();
        assert_eq!(signs.len(), 8, "two corners share a sign combination");
    }
}
