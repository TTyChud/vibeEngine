//! Compiles the engine's built-in shaders to SPIR-V and validates the result
//! with `spirv-val`, so the output is checked by something outside this crate.
//!
//! Run with: cargo run -p vibe-shader --example compile

use vibe_shader::{
    CompileOptions, ShaderError, compile_entry_point, looks_like_spirv, words_to_bytes,
};

/// The 2D batched quad shader: one draw call, quads addressed by instance.
const QUAD: &str = r#"
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
    let texel = textureSample(textures, tex_sampler, in.uv);
    return texel * in.color;
}
"#;

/// A compute pass that resolves a physics broad phase into a draw list.
const CULL: &str = r#"
struct DrawItem {
    position: vec2<f32>,
    radius: f32,
}

@group(0) @binding(0) var<storage, read> items: array<DrawItem>;
@group(0) @binding(1) var<storage, read_write> visible: array<u32>;
@group(0) @binding(2) var<uniform> view: vec4<f32>;

@compute @workgroup_size(64)
fn cs_main(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.x;
    if (index >= arrayLength(&items)) {
        return;
    }
    let item = items[index];
    let centre = view.xy;
    let extent = view.zw;
    let inside = abs(item.position.x - centre.x) < extent.x + item.radius
        && abs(item.position.y - centre.y) < extent.y + item.radius;
    if (inside) {
        visible[index] = index;
    }
}
"#;

fn report(name: &str, source: &str, entry: (&str, naga::ShaderStage)) {
    print!("{name}: ");
    match compile_entry_point(source, CompileOptions::release(), Some(entry)) {
        Ok(words) => {
            let bytes = words_to_bytes(&words);
            println!(
                "{} words ({} bytes), magic ok: {}",
                words.len(),
                bytes.len(),
                looks_like_spirv(&words)
            );

            // Hand the bytes to spirv-val so an external tool agrees the module
            // is real, rather than trusting this crate's own check.
            let path = std::env::temp_dir().join(format!("vibe-{name}.spv"));
            if std::fs::write(&path, &bytes).is_ok() {
                match std::process::Command::new("spirv-val").arg(&path).output() {
                    Ok(out) if out.status.success() => println!("  spirv-val: OK"),
                    Ok(out) => println!(
                        "  spirv-val: FAILED\n{}",
                        String::from_utf8_lossy(&out.stderr)
                    ),
                    Err(e) => println!("  spirv-val: not installed ({e})"),
                }
                let _ = std::fs::remove_file(&path);
            }
        }
        Err(ShaderError::Parse(m)) => println!("parse error: {m}"),
        Err(ShaderError::Validation { message, .. }) => println!("validation error: {message}"),
        Err(ShaderError::Backend(m)) => println!("backend error: {m}"),
        Err(e) => println!("error: {e}"),
    }
}

fn main() {
    report("quad.vert", QUAD, ("vs_main", naga::ShaderStage::Vertex));
    report("quad.frag", QUAD, ("fs_main", naga::ShaderStage::Fragment));
    report("cull.comp", CULL, ("cs_main", naga::ShaderStage::Compute));
}
