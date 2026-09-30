//! Loads a glTF or GLB file and reports what is in it.
//!
//! Run with: cargo run -p vibe-gltf --example inspect -- path/to/model.glb
//!
//! With no argument it builds a small model in memory, so the loader's own
//! output can be checked without needing an asset on disk.

use std::path::Path;

use vibe_gltf::{Mesh, MeshVertex, Model, Playback, SheetClock, SheetDesc, slice};

fn main() {
    let Some(arg) = std::env::args().nth(1) else {
        println!("no file given; reporting a synthetic model instead\n");
        report(&synthetic(), "synthetic");
        return;
    };
    let path = Path::new(&arg);
    match vibe_gltf::load(path) {
        Ok(model) => report(&model, &arg),
        Err(e) => {
            eprintln!("FAIL loading {arg}: {e}");
            std::process::exit(1);
        }
    }
}

fn report(model: &Model, name: &str) {
    println!("model: {name}");
    println!("  meshes:      {}", model.meshes.len());
    println!("  vertices:    {}", model.vertex_count());
    println!("  triangles:   {}", model.triangle_count());
    println!("  skins:       {}", model.skins.len());
    println!("  materials:   {}", model.materials.len());
    println!("  images:      {}", model.images.len());
    println!("  nodes:       {}", model.node_transforms.len());
    println!("  clips:       {}", model.clips.len());

    for (i, m) in model.materials.iter().enumerate() {
        let tex = model
            .material_textures
            .get(i)
            .and_then(|t| *t)
            .map(|t| format!("texture {} uv{}", t.index, t.tex_coord))
            .unwrap_or_else(|| "no base colour texture".to_string());
        println!(
            "  material {i}: base {:?} metallic {} roughness {} {tex}",
            m.base_color, m.metallic, m.roughness
        );
    }

    for (i, s) in model.skins.iter().enumerate() {
        println!("  skin {i}: {} joint(s)", s.len());
    }

    for (i, c) in model.clips.iter().enumerate() {
        println!(
            "  clip {i}: {:?} {} channel(s), {:.2}s, channels:",
            c.name,
            c.channels.len(),
            c.duration()
        );
        for ch in &c.channels {
            let end = ch.sample(ch.duration());
            let mid = ch.sample(ch.duration() * 0.5);
            println!(
                "    node {} {}: {} keyframe(s), midpoint {:?}, end {:?}",
                ch.node,
                ch.path.gltf_name(),
                ch.keyframes.len(),
                mid,
                end
            );
        }
    }

    if let Some((lo, hi)) = model.bounds() {
        println!("  bounds:      {lo:?} .. {hi:?}");
    }

    sheet();

    if model.meshes.iter().any(|m| !m.is_empty()) {
        let mut playback = Playback::new(model.longest_clip().unwrap_or(0));
        if let Some(clip) = model.clip(playback.clip_index) {
            let sample = playback.advance(clip, 0.1);
            println!("  playback:    t=0.1 -> wrapped {sample:.3}");
        }
    }
}

/// A model built by hand, so the reporting path is exercised with no file.
fn synthetic() -> Model {
    let vertices: Vec<MeshVertex> = (0..3)
        .map(|i| MeshVertex {
            position: [i as f32, 0.0, 0.0],
            normal: [0.0, 0.0, 1.0],
            uv: [0.0, 0.0],
            joints: [0; 4],
            weights: [1.0, 0.0, 0.0, 0.0],
        })
        .collect();
    Model {
        name: "synthetic".to_string(),
        meshes: vec![Mesh {
            vertices,
            indices: vec![0, 1, 2],
            material: 0,
            skin: None,
        }],
        materials: vec![vibe_gltf::Material::default()],
        ..Default::default()
    }
}

/// A sprite sheet sliced and timed, to show the sheet path working.
fn sheet() {
    let desc = SheetDesc::grid(4, 2).with_fps(12.0);
    let frames = slice(&desc, 128, 64);
    let mut clock = SheetClock::new(&desc, frames.len());
    println!(
        "\nsheet: {} frame(s) at {:.3}s each",
        frames.len(),
        desc.frame_seconds
    );
    for _ in 0..4 {
        let r = frames[clock.frame()];
        println!(
            "  frame {}: {}x{}+{}+{}",
            clock.frame(),
            r.width,
            r.height,
            r.x,
            r.y
        );
        clock.advance(desc.frame_seconds);
    }
}
