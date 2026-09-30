//! Builds an MSDF atlas from a system font and writes it as a PPM.
//!
//! The point is to look at it: a distance field that is wrong in a subtle way
//! — the wrong sign, a half-pixel offset, a transposed bitmap — still produces
//! an atlas, and only a picture shows which one it is.
//!
//! Run with: cargo run -p vibe-gltf --example atlas -- [font.ttf] [out.ppm]

use vibe_gltf::{CHANNELS, MsdfFont, layout};

/// Fonts to try, in order, when none is named.
///
/// Hard-coding one path means the example silently does nothing on a machine
/// without that font, which looks exactly like the atlas working.
const CANDIDATES: &[&str] = &[
    "/usr/share/fonts/noto/NotoSans-Regular.ttf",
    "/usr/share/fonts/TTF/DejaVuSans.ttf",
    "/usr/share/fonts/gnu-free/FreeSans.ttf",
    "/usr/share/fonts/TTF/LiberationSans-Regular.ttf",
];

fn find_font() -> Option<String> {
    CANDIDATES
        .iter()
        .find(|p| std::path::Path::new(p).exists())
        .map(|p| (*p).to_string())
}

fn main() {
    let mut args = std::env::args().skip(1);
    let Some(font_path) = args.next().or_else(find_font) else {
        eprintln!("no font given and none of the known paths exist: {CANDIDATES:?}");
        std::process::exit(1);
    };
    let out = args
        .next()
        .unwrap_or_else(|| "/tmp/vibe-msdf.ppm".to_string());
    let text = "ABCDEFGHIJKLM abcdefghijklm 0123456789";

    let data = match std::fs::read(&font_path) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("FAIL reading {font_path}: {e}");
            std::process::exit(1);
        }
    };

    let start = std::time::Instant::now();
    let font = match MsdfFont::new(&data, text, 48.0, 4) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("FAIL building the atlas: {e}");
            std::process::exit(1);
        }
    };
    let elapsed = start.elapsed();

    let a = &font.atlas;
    println!(
        "atlas {}x{} for {} glyph(s) in {:?}",
        a.width,
        a.height,
        a.glyphs.len(),
        elapsed
    );
    println!(
        "  {} px, {} channels, range {}",
        a.pixels.len(),
        CHANNELS,
        a.distance_range()
    );
    println!("  em size {}, px per em {}", a.units_per_em, a.px_per_em);

    // The field's own statistics are the check: a real field has a spread of
    // values around the midpoint, an all-zero one means nothing was written, and
    // an all-255 one means every glyph is solid.
    let mut min = 255u8;
    let mut max = 0u8;
    let mut sum = 0u64;
    for p in &a.pixels {
        min = min.min(*p);
        max = max.max(*p);
        sum += *p as u64;
    }
    let mean = sum as f32 / a.pixels.len().max(1) as f32;
    println!("  field range {min}..{max}, mean {mean:.1}");
    if max == 0 {
        eprintln!("WARNING: the field is empty, so no glyph was rasterised");
    }

    // Write the atlas as a PPM. The three channels are the three distances, so
    // this is the raw field, not a rendered glyph: R shows one edge's distance,
    // and the three together are what the shader takes the median of.
    let mut ppm = format!("P6\n{} {}\n255\n", a.width, a.height).into_bytes();
    for px in a.pixels.chunks_exact(CHANNELS) {
        ppm.extend_from_slice(&px[..3]);
    }
    if let Err(e) = std::fs::write(&out, &ppm) {
        eprintln!("FAIL writing {out}: {e}");
        std::process::exit(1);
    }
    println!("  wrote {out}");

    // A laid-out run, so the metrics are visible too.
    let run = layout(a, "Hello", 48.0);
    println!("\nlayout of \"Hello\" at 48px:");
    for g in &run.glyphs {
        println!(
            "  {:?} at ({:.1}, {:.1}) advance {:.1} uv [{:.3}..{:.3}]x[{:.3}..{:.3}]",
            g.character,
            g.position.x,
            g.position.y,
            g.advance,
            g.uvs[0][0],
            g.uvs[1][0],
            g.uvs[2][1],
            g.uvs[0][1],
        );
    }
    println!("  width {:.1}, height {:.1}", run.width, run.height);
}
