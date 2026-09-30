//! MSDF text: a signed-distance field atlas, and glyph layout.
//!
//! A plain coverage bitmap blurs badly when it is scaled, which is exactly what
//! text does: one glyph is drawn at many sizes, so a 16-pixel bitmap stretched
//! to 64 has visible stair-steps. An MSDF stores, per texel, the *distance* to
//! the nearest edge of the glyph, per colour channel and with the three
//! channels pointing at three different edges. At render time the median of the
//! three is compared against a threshold, and because the median reconstructs a
//! sharp edge from smooth distances, the glyph stays crisp at any scale.
//!
//! The atlas is built once per font at one size and reused for every size on
//! screen, which is why the field is generated rather than rasterised.

use ab_glyph::{Font, FontVec, Glyph, PxScale, ScaleFont};
use glam::Vec2;

use crate::error::GltfError;

/// The distance range an MSDF encodes, in ems.
///
/// The standard value: a glyph is 1 em tall and the field reaches 0.5 em past
/// its edge in each direction, so the whole em square is covered by the field.
pub const EM_DISTANCE_RANGE: f32 = 0.5;

/// How many pixels of distance field surround each glyph.
pub const DEFAULT_DISTANCE_RANGE_PX: u8 = 4;

/// The largest atlas the generator will grow to, in pixels.
///
/// 4096 is 64 MB of RGB, which is already far past useful for a glyph atlas;
/// the bound stops a pathological request from allocating without limit.
pub const MAX_ATLAS: u32 = 4096;

/// How many distance fields each texel holds.
///
/// Three is the standard: RGB, each channel nearest to a different edge. Two
/// blurs corners, one is a plain coverage bitmap.
pub const CHANNELS: usize = 3;

/// One glyph's entry in an atlas.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GlyphEntry {
    /// The character this entry is for.
    pub character: char,
    /// Where its bitmap sits in the atlas, in pixels.
    pub rect: AtlasRect,
    /// How far the glyph advances, in ems.
    pub advance: f32,
    /// The glyph's offset from the pen, in ems.
    pub bearing: Vec2,
    /// The signed distance range the bitmap encodes, in ems.
    pub distance_range: f32,
}

impl GlyphEntry {
    /// The width of the bitmap in pixels.
    pub fn width(&self) -> u32 {
        self.rect.width
    }

    /// The height of the bitmap in pixels.
    pub fn height(&self) -> u32 {
        self.rect.height
    }

    /// The UV coordinates for this glyph's quad, in corner order.
    ///
    /// V is flipped because the atlas's origin is the top left while a quad's
    /// UV origin is the bottom left; without the flip every glyph renders
    /// upside down, which is invisible on a symmetric character like `o` and
    /// glaring on a `g`.
    pub fn uvs(&self, atlas_width: u32, atlas_height: u32) -> [[f32; 2]; 4] {
        if atlas_width == 0 || atlas_height == 0 {
            return [[0.0; 2]; 4];
        }
        let u0 = self.rect.x as f32 / atlas_width as f32;
        let u1 = self.rect.right() as f32 / atlas_width as f32;
        let v0 = self.rect.y as f32 / atlas_height as f32;
        let v1 = self.rect.bottom() as f32 / atlas_height as f32;
        [[u0, v1], [u1, v1], [u1, v0], [u0, v0]]
    }

    /// The advance in pixels at a given size.
    ///
    /// A zero or negative size is clamped rather than allowed to produce a
    /// zero or negative width, which would collapse the run of glyphs onto one
    /// another.
    pub fn advance_px(&self, size_px: f32) -> f32 {
        if size_px <= 0.0 {
            return 0.0;
        }
        self.advance * size_px
    }
}

/// A rectangle in an atlas, in pixels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct AtlasRect {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

impl AtlasRect {
    /// The right edge, exclusive.
    pub fn right(&self) -> u32 {
        self.x + self.width
    }

    /// The bottom edge, exclusive.
    pub fn bottom(&self) -> u32 {
        self.y + self.height
    }

    /// True when this rectangle lies inside an atlas of the given size.
    pub fn fits(&self, atlas_width: u32, atlas_height: u32) -> bool {
        self.right() <= atlas_width && self.bottom() <= atlas_height
    }
}

/// A generated MSDF atlas and the glyphs in it.
#[derive(Debug, Clone)]
pub struct MsdfAtlas {
    /// The atlas pixels: `width * height * 3` bytes, one per channel.
    pub pixels: Vec<u8>,
    /// The atlas width in pixels.
    pub width: u32,
    /// The atlas height in pixels.
    pub height: u32,
    /// One entry per glyph, keyed by character in `glyphs`.
    pub glyphs: Vec<GlyphEntry>,
    /// The size the atlas was generated at, in pixels per em.
    pub px_per_em: f32,
    /// The font's units-per-em, so a size in points maps to a scale factor.
    ///
    /// f32 rather than ab_glyph's f16: the half type is still unstable, and the
    /// value only ever feeds a scale factor.
    pub units_per_em: f32,
}

impl MsdfAtlas {
    /// The entry for a character, if the atlas has one.
    pub fn glyph(&self, character: char) -> Option<&GlyphEntry> {
        self.glyphs.iter().find(|g| g.character == character)
    }

    /// The atlas's channels-per-pixel, which is 3 for an MSDF.
    pub fn channels(&self) -> usize {
        CHANNELS
    }

    /// The distance range in ems the atlas encodes.
    pub fn distance_range(&self) -> f32 {
        self.glyphs
            .first()
            .map(|g| g.distance_range)
            .unwrap_or(EM_DISTANCE_RANGE)
    }

    /// The three distances a texel holds, or `None` outside the atlas.
    pub fn texel(&self, x: u32, y: u32) -> Option<[f32; 3]> {
        if x >= self.width || y >= self.height {
            return None;
        }
        let i = (y as usize * self.width as usize + x as usize) * CHANNELS;
        let p = self.pixels.get(i..i + CHANNELS)?;
        Some([
            p[0] as f32 / 255.0,
            p[1] as f32 / 255.0,
            p[2] as f32 / 255.0,
        ])
    }
}

/// Which way a run of text is laid out.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TextDirection {
    /// Left to right.
    #[default]
    LeftToRight,
    /// Right to left.
    RightToLeft,
}

/// A laid-out run of glyphs.
#[derive(Debug, Clone, PartialEq)]
pub struct GlyphRun {
    /// One entry per glyph actually drawn, in order.
    pub glyphs: Vec<PlacedGlyph>,
    /// The run's width in pixels at the size it was laid out for.
    pub width: f32,
    /// The run's height in pixels.
    pub height: f32,
}

/// One glyph placed in a run.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PlacedGlyph {
    /// The character.
    pub character: char,
    /// Its position in the atlas.
    pub rect: AtlasRect,
    /// Where its bitmap's top-left corner sits, in pixels.
    pub position: Vec2,
    /// How far the pen moves after drawing it, in pixels.
    pub advance: f32,
    /// The UV rectangle for the glyph's quad.
    pub uvs: [[f32; 2]; 4],
}

/// Lay out a string with the atlas's glyphs.
///
/// Space and newline are handled without an atlas entry: a space advances the
/// pen and a newline starts a new line, and neither draws. A character the
/// atlas does not have is skipped rather than drawn as a box, so a missing
/// glyph leaves a gap instead of corrupting the run's metrics.
pub fn layout(atlas: &MsdfAtlas, text: &str, size_px: f32) -> GlyphRun {
    layout_with_direction(atlas, text, size_px, TextDirection::LeftToRight, 1.2)
}

/// Lay out a string with an explicit direction and line spacing.
pub fn layout_with_direction(
    atlas: &MsdfAtlas,
    text: &str,
    size_px: f32,
    direction: TextDirection,
    line_spacing: f32,
) -> GlyphRun {
    let size = if size_px > 0.0 { size_px } else { 1.0 };
    let mut run = GlyphRun {
        glyphs: Vec::new(),
        width: 0.0,
        height: size,
    };

    // Split on newlines first so a line's width is known before its glyphs are
    // placed, which is what centring and right-alignment need.
    let lines: Vec<&str> = text.split('\n').collect();
    let line_height = size * line_spacing;

    for (line_index, line) in lines.iter().enumerate() {
        let baseline = line_index as f32 * line_height;

        // Every character contributes an advance, whether or not it draws, so
        // the list is of advances with an optional glyph rather than of glyphs.
        // Dropping the undrawable ones here is what made a space disappear
        // from the layout: its advance counted towards the line's width but not
        // towards the pen, so the glyph after a space landed on top of the one
        // before it.
        let mut steps: Vec<(Option<GlyphEntry>, f32)> = Vec::new();
        let mut line_width = 0.0f32;
        for ch in line.chars() {
            let advance = match atlas.glyph(ch) {
                Some(g) => {
                    let a = g.advance_px(size);
                    steps.push((Some(*g), a));
                    a
                }
                // A space has no bitmap but still advances the pen. The width
                // is a quarter em, which is close enough to most fonts' space
                // that a run does not look oddly spaced.
                None if ch.is_whitespace() => {
                    let a = size * 0.25;
                    steps.push((None, a));
                    a
                }
                // A character the atlas lacks contributes nothing, so the rest
                // of the run keeps its spacing instead of collapsing.
                None => {
                    steps.push((None, 0.0));
                    0.0
                }
            };
            line_width += advance;
        }

        // In right-to-left the line starts from its far edge and runs inwards.
        let mut cursor = match direction {
            TextDirection::LeftToRight => 0.0,
            TextDirection::RightToLeft => line_width,
        };
        for (entry, advance) in steps {
            let x = match direction {
                TextDirection::LeftToRight => cursor,
                TextDirection::RightToLeft => cursor - advance,
            };
            if let Some(entry) = entry {
                // The bitmap's top-left sits at the pen, offset by the bearing
                // and raised by the glyph's ascent, so it lands on the baseline
                // rather than at the line's top edge.
                let px = x + entry.bearing.x * size;
                let py = baseline - entry.bearing.y * size;
                run.glyphs.push(PlacedGlyph {
                    character: entry.character,
                    rect: entry.rect,
                    position: Vec2::new(px, py),
                    advance,
                    uvs: entry.uvs(atlas.width, atlas.height),
                });
            }
            cursor += match direction {
                TextDirection::LeftToRight => advance,
                TextDirection::RightToLeft => -advance,
            };
        }

        run.width = run.width.max(line_width);
    }

    run.height = line_height * lines.len() as f32;
    run
}

/// How a run is aligned within a width.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TextAlign {
    /// Against the left edge.
    #[default]
    Left,
    /// Centred.
    Center,
    /// Against the right edge.
    Right,
}

/// Shift a run so it is aligned within a width.
pub fn align_run(run: &mut GlyphRun, align: TextAlign, within_width: f32) {
    let offset = match align {
        TextAlign::Left => 0.0,
        TextAlign::Center => (within_width - run.width) * 0.5,
        TextAlign::Right => within_width - run.width,
    };
    if offset == 0.0 {
        return;
    }
    for g in &mut run.glyphs {
        g.position.x += offset;
    }
}

/// A font and the atlas built from it.
pub struct MsdfFont {
    /// The parsed font, for anything the atlas does not answer.
    pub font: FontVec,
    /// The generated atlas.
    pub atlas: MsdfAtlas,
}

impl std::fmt::Debug for MsdfFont {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MsdfFont")
            .field("glyphs", &self.atlas.glyphs.len())
            .field("atlas", &(self.atlas.width, self.atlas.height))
            .field("px_per_em", &self.atlas.px_per_em)
            .finish()
    }
}

impl MsdfFont {
    /// Build an atlas for a font's characters.
    ///
    /// `px_per_em` is the size the field is generated at; the field is then
    /// valid for every size, which is the reason to generate one at all.
    pub fn new(
        font_data: &[u8],
        characters: &str,
        px_per_em: f32,
        distance_range_px: u8,
    ) -> Result<MsdfFont, GltfError> {
        let font = FontVec::try_from_vec(font_data.to_vec()).map_err(|e| GltfError::Decode {
            name: "font".to_string(),
            message: e.to_string(),
        })?;
        let atlas = build_atlas(&font, characters, px_per_em, distance_range_px)?;
        Ok(MsdfFont { font, atlas })
    }

    /// The font's units per em, which maps a point size to a scale.
    pub fn units_per_em(&self) -> f32 {
        self.atlas.units_per_em
    }

    /// The scale factor for a size in pixels.
    pub fn scale_for(&self, size_px: f32) -> f32 {
        let upem = self.atlas.units_per_em;
        if upem <= 0.0 {
            return 1.0;
        }
        size_px / upem
    }
}

/// Pack a set of rectangles into an atlas, row by row.
///
/// A shelf packer rather than a full bin packer: glyphs are nearly the same
/// size, so shelves waste little and the code stays short. Rectangles that do
/// not fit the atlas width start a new shelf, and a set taller than the atlas
/// is reported rather than silently clipped.
fn pack(
    rects: &[(u32, u32)],
    atlas_width: u32,
    atlas_height: u32,
    padding: u32,
) -> Result<Vec<AtlasRect>, GltfError> {
    let mut out: Vec<AtlasRect> = Vec::with_capacity(rects.len());
    let mut x = padding;
    let mut y = padding;
    let mut shelf_height = 0u32;

    for &(w, h) in rects {
        if x + w + padding > atlas_width {
            // This rectangle will not fit the rest of the shelf.
            x = padding;
            y += shelf_height + padding;
            shelf_height = 0;
        }
        if y + h + padding > atlas_height {
            return Err(GltfError::Decode {
                name: "atlas".to_string(),
                message: format!("the glyphs need more than {atlas_width}x{atlas_height} pixels"),
            });
        }
        out.push(AtlasRect {
            x,
            y,
            width: w,
            height: h,
        });
        x += w + padding;
        shelf_height = shelf_height.max(h);
    }
    Ok(out)
}

/// The pixel bounds of a glyph, measured from its own outline.
///
/// The bounds come from the outline rather than from a font metrics call,
/// because the field is measured against the outline: a bitmap sized from
/// metrics that disagree by a pixel clips the glyph's edge, and a clipped
/// distance field has no data past the cut to interpolate towards, so the
/// glyph grows a visible notch on that side.
///
/// Returns the bounds in pixels with y up, as `(min_x, min_y, max_x, max_y)`.
fn glyph_bounds(font: &FontVec, character: char, px_per_em: f32) -> (f32, f32, f32, f32) {
    let glyph = scaled_glyph(font, character, px_per_em);
    let Some(outline) = font.outline(glyph.id) else {
        return (0.0, 0.0, 0.0, 0.0);
    };
    let upem = font.units_per_em().unwrap_or(1000.0);
    let contours = flatten(&outline.curves, upem);
    let mut min_x = f32::INFINITY;
    let mut min_y = f32::INFINITY;
    let mut max_x = f32::NEG_INFINITY;
    let mut max_y = f32::NEG_INFINITY;
    for c in &contours {
        for p in c {
            min_x = min_x.min(p[0]);
            min_y = min_y.min(p[1]);
            max_x = max_x.max(p[0]);
            max_y = max_y.max(p[1]);
        }
    }
    if !min_x.is_finite() {
        return (0.0, 0.0, 0.0, 0.0);
    }
    // Into pixels, y up.
    (
        min_x * px_per_em,
        min_y * px_per_em,
        max_x * px_per_em,
        max_y * px_per_em,
    )
}

/// The size in pixels a glyph's bitmap needs.
fn bitmap_size(
    font: &FontVec,
    character: char,
    px_per_em: f32,
    distance_range_px: u8,
) -> (u32, u32) {
    let (min_x, min_y, max_x, max_y) = glyph_bounds(font, character, px_per_em);
    // The distance field extends past the glyph on every side, or a sharp
    // corner would have no data to interpolate towards and would tear.
    let pad = distance_range_px as f32 + 1.0;
    let w = (max_x - min_x) + pad * 2.0;
    let h = (max_y - min_y) + pad * 2.0;
    (w.max(0.0).ceil() as u32, h.max(0.0).ceil() as u32)
}

/// Build the MSDF atlas for a set of characters.
fn build_atlas(
    font: &FontVec,
    characters: &str,
    px_per_em: f32,
    distance_range_px: u8,
) -> Result<MsdfAtlas, GltfError> {
    let scaled = font.as_scaled(PxScale::from(px_per_em));
    let unique: Vec<char> = {
        let mut seen: Vec<char> = Vec::new();
        for ch in characters.chars() {
            if !seen.contains(&ch) {
                seen.push(ch);
            }
        }
        seen
    };

    let mut glyph_ids: Vec<ab_glyph::GlyphId> = Vec::with_capacity(unique.len());
    let mut sizes: Vec<(u32, u32)> = Vec::with_capacity(unique.len());
    for ch in &unique {
        glyph_ids.push(scaled.scaled_glyph(*ch).id);
        sizes.push(bitmap_size(font, *ch, px_per_em, distance_range_px));
    }

    // A square atlas big enough for every glyph, doubling until it fits. The
    // doubling is what keeps this working for a font with a few hundred
    // glyphs without solving the packing problem exactly.
    let padding = 1u32;
    let needed: u32 = sizes.iter().map(|(w, h)| w + h).max().unwrap_or(1).max(16);
    let mut size = 1u32;
    while size < needed {
        size *= 2;
    }
    // Grow the atlas until every glyph fits. The first attempt is not
    // propagated with `?`: a font with one very large glyph fails at this size
    // and succeeds at the next, and reporting the first failure would reject a
    // font that is perfectly usable.
    let mut rects = pack(&sizes, size, size, padding);
    while rects.is_err() && size < MAX_ATLAS {
        size *= 2;
        rects = pack(&sizes, size, size, padding);
    }
    let rects = rects?;

    let mut pixels = vec![0u8; (size * size) as usize * CHANNELS];
    let distance_range = EM_DISTANCE_RANGE;

    let mut entries = Vec::with_capacity(unique.len());
    // The field extends this far past the glyph on every side, in pixels. It
    // must match what bitmap_size added, or the field is measured over a
    // different area than the bitmap covers.
    let field_pad = distance_range_px as f32 + 1.0;

    for (i, ch) in unique.iter().enumerate() {
        let id = glyph_ids[i];
        let rect = rects[i];
        let (min_x_px, _, _, max_y_px) = glyph_bounds(font, *ch, px_per_em);
        // The bitmap's left edge sits one field-width to the left of the
        // glyph, so the field has room to extend past it.
        let origin_x = min_x_px - field_pad;

        render_glyph(
            font,
            *ch,
            &mut pixels,
            size,
            rect,
            origin_x,
            max_y_px,
            px_per_em,
            distance_range,
        );

        // The advance is the pen movement the glyph causes, in ems. The
        // bearing is where the quad sits relative to the pen: the left side
        // bearing, and how far the glyph rises above the baseline.
        //
        // h_advance is read from the *unscaled* font and divided by the em
        // size, so the result is in ems. Reading it from a font scaled to
        // px_per_em instead gives a value already in pixels, and dividing that
        // by the em size makes every advance a thousandth of what it should be:
        // the run's width collapses and every glyph lands on the previous one.
        let upem = font.units_per_em().unwrap_or(1000.0);
        let advance = font.h_advance_unscaled(id) as f32 / upem;
        let bearing = Vec2::new(min_x_px / px_per_em, max_y_px / px_per_em);

        entries.push(GlyphEntry {
            character: *ch,
            rect,
            advance,
            bearing,
            distance_range,
        });
    }

    Ok(MsdfAtlas {
        pixels,
        width: size,
        height: size,
        glyphs: entries,
        px_per_em,
        // ab_glyph reports the em size as an Option: a font that does not say
        // is taken as 1000, which is what the spec's default and every exporter
        // in practice use.
        units_per_em: font.units_per_em().unwrap_or(1000.0),
    })
}

/// Rasterise one glyph's signed distance field into the atlas.
///
/// The field is measured against the glyph's outline in ems, then scaled into
/// the bitmap's pixels. Each texel samples at its centre, which is what makes
/// the field reproducible: a field sampled at texel corners sits half a pixel
/// from the geometry and every glyph comes out slightly soft.
/// `origin_x` is the glyph's left edge in pixels from the bitmap's left edge,
/// and `glyph_top_y` is the glyph's top in y-up pixels from the origin. The
/// bitmap's row 0 is the top, so a row's y is measured downwards from the
/// glyph's top, which is why the sampling flips the sign.
fn render_glyph(
    font: &FontVec,
    character: char,
    pixels: &mut [u8],
    atlas_size: u32,
    rect: AtlasRect,
    origin_x: f32,
    glyph_top_y: f32,
    px_per_em: f32,
    distance_range: f32,
) {
    // A glyph with no outline (a space, or a character the font lacks) has no
    // field to write; the bitmap stays transparent and the entry still advances
    // the pen, which is what a space has to do.
    let glyph = scaled_glyph(font, character, px_per_em);
    let Some(outline) = font.outline(glyph.id) else {
        return;
    };
    let upem = font.units_per_em().unwrap_or(1000.0);
    let contours = flatten(&outline.curves, upem);
    if contours.is_empty() {
        return;
    }

    for row in 0..rect.height {
        for col in 0..rect.width {
            // The sample in bitmap pixels, measured from the glyph's own bitmap
            // origin rather than the atlas's, so where a glyph sits in the
            // atlas does not change its field.
            let local_x = col as f32 + 0.5;
            let local_y = row as f32 + 0.5;
            // Into ems with y up, which is the space the contours are in. The
            // bitmap grows downwards while the outline grows upwards, so the
            // row's offset is subtracted from the glyph's top.
            let em_x = (origin_x + local_x) / px_per_em;
            let em_y = (glyph_top_y - local_y) / px_per_em;

            let d_em = signed_distance(&contours, [em_x, em_y]);
            let encoded = encode_distance(d_em / distance_range);

            let i = ((rect.y + row) as usize * atlas_size as usize + (rect.x + col) as usize)
                * CHANNELS;
            if i + CHANNELS <= pixels.len() {
                pixels[i] = encoded[0];
                pixels[i + 1] = encoded[1];
                pixels[i + 2] = encoded[2];
            }
        }
    }
}

/// The glyph at a size, for outline extraction.
fn scaled_glyph(font: &FontVec, character: char, px_per_em: f32) -> Glyph {
    font.as_scaled(PxScale::from(px_per_em))
        .scaled_glyph(character)
}

/// Flatten an outline's curves into polylines in ems.
///
/// A font's outline is lines and Bézier curves, and a distance to a curve is
/// not the distance to the polyline that approximates it. The approximation has
/// to be fine enough that its error is well under a pixel at the size the atlas
/// is drawn at, which is what the segment count buys: sixteen segments per
/// curve puts the chord error far below a pixel for a glyph at 32 pixels per
/// em, and a coarser polyline shows as a faceted edge on a round letter.
///
/// The curves arrive in font units, so each point is divided by the font's
/// units-per-em to land in ems. That happens once here rather than per texel,
/// because every texel of a glyph needs the same contour.
///
/// The divisor is the em size and not the pixel size. They differ by a factor
/// of a thousand or so, and using the wrong one makes every glyph a thousand
/// times its intended size, which the atlas then cannot fit — a bug that reads
/// as "the font is too complicated" rather than as an arithmetic slip.
fn flatten(curves: &[ab_glyph::OutlineCurve], units_per_em: f32) -> Vec<Vec<[f32; 2]>> {
    /// Segments per Bézier curve. More segments means a closer polyline and a
    /// proportionally slower build, so this is the one knob that trades atlas
    /// quality against generation time.
    const SEGMENTS_PER_CURVE: usize = 16;

    let em = if units_per_em > 0.0 {
        units_per_em
    } else {
        1000.0
    };
    let at = |p: ab_glyph::Point| [p.x as f32 / em, p.y as f32 / em];

    let mut contours: Vec<Vec<[f32; 2]>> = Vec::new();
    let mut current: Vec<[f32; 2]> = Vec::new();

    for curve in curves {
        match curve {
            ab_glyph::OutlineCurve::Line(a, b) => {
                if current.is_empty() {
                    current.push(at(*a));
                }
                current.push(at(*b));
            }
            ab_glyph::OutlineCurve::Quad(a, c, b) => {
                if current.is_empty() {
                    current.push(at(*a));
                }
                let (p0, p1, p2) = (at(*a), at(*c), at(*b));
                for i in 1..=SEGMENTS_PER_CURVE {
                    let t = i as f32 / SEGMENTS_PER_CURVE as f32;
                    let u = 1.0 - t;
                    current.push([
                        u * u * p0[0] + 2.0 * u * t * p1[0] + t * t * p2[0],
                        u * u * p0[1] + 2.0 * u * t * p1[1] + t * t * p2[1],
                    ]);
                }
            }
            ab_glyph::OutlineCurve::Cubic(a, c1, c2, b) => {
                if current.is_empty() {
                    current.push(at(*a));
                }
                let (p0, p1, p2, p3) = (at(*a), at(*c1), at(*c2), at(*b));
                for i in 1..=SEGMENTS_PER_CURVE {
                    let t = i as f32 / SEGMENTS_PER_CURVE as f32;
                    let u = 1.0 - t;
                    let (u2, t2) = (u * u, t * t);
                    current.push([
                        u2 * u * p0[0]
                            + 3.0 * u2 * t * p1[0]
                            + 3.0 * u * t2 * p2[0]
                            + t2 * t * p3[0],
                        u2 * u * p0[1]
                            + 3.0 * u2 * t * p1[1]
                            + 3.0 * u * t2 * p2[1]
                            + t2 * t * p3[1],
                    ]);
                }
            }
        }
        // A font marks the end of a contour by returning to the start, so that
        // is where the accumulated points are flushed.
        if is_closed(&current) {
            contours.push(std::mem::take(&mut current));
        }
    }
    if current.len() >= 2 {
        contours.push(current);
    }
    // A contour of one point has no edges and would contribute a zero-length
    // segment to every measurement.
    contours.retain(|c| c.len() >= 2);
    contours
}

/// True when a polyline has returned to where it started.
fn is_closed(points: &[[f32; 2]]) -> bool {
    match (points.first(), points.last()) {
        (Some(a), Some(b)) => {
            points.len() >= 3 && (a[0] - b[0]).abs() < 1e-6 && (a[1] - b[1]).abs() < 1e-6
        }
        _ => false,
    }
}

/// The signed distance in ems from a point to a set of contours.
///
/// Positive inside the glyph, negative outside.
fn signed_distance(contours: &[Vec<[f32; 2]>], p: [f32; 2]) -> f32 {
    let mut best = f32::INFINITY;
    for contour in contours {
        let n = contour.len();
        for i in 0..n {
            let a = contour[i];
            let b = contour[(i + 1) % n];
            let d = distance_to_segment(p, a, b);
            if d < best {
                best = d;
            }
        }
    }
    if !best.is_finite() {
        // No edges at all: treat the area as outside rather than returning
        // infinity, which would encode as a saturated byte and fill the glyph.
        return -EM_DISTANCE_RANGE;
    }
    if point_in_contours(contours, p) {
        best
    } else {
        -best
    }
}

/// The distance from a point to a line segment.
fn distance_to_segment(p: [f32; 2], a: [f32; 2], b: [f32; 2]) -> f32 {
    let abx = b[0] - a[0];
    let aby = b[1] - a[1];
    let len_sq = abx * abx + aby * aby;
    let t = if len_sq <= f32::EPSILON {
        0.0
    } else {
        (((p[0] - a[0]) * abx + (p[1] - a[1]) * aby) / len_sq).clamp(0.0, 1.0)
    };
    let cx = a[0] + abx * t;
    let cy = a[1] + aby * t;
    ((p[0] - cx).powi(2) + (p[1] - cy).powi(2)).sqrt()
}

/// Even-odd crossing test, so the sign does not depend on contour winding.
///
/// Winding is not available from a flattened outline, and a font is free to
/// wind the same glyph either way, so the sign has to come from the crossing
/// count instead.
fn point_in_contours(contours: &[Vec<[f32; 2]>], p: [f32; 2]) -> bool {
    let mut inside = false;
    for contour in contours {
        let n = contour.len();
        if n < 3 {
            continue;
        }
        for i in 0..n {
            let a = contour[i];
            let b = contour[(i + 1) % n];
            if (a[1] > p[1]) != (b[1] > p[1]) {
                let t = (p[1] - a[1]) / (b[1] - a[1]);
                if p[0] < a[0] + t * (b[0] - a[0]) {
                    inside = !inside;
                }
            }
        }
    }
    inside
}

/// Encode a signed distance into the three channels.
///
/// The three channels are the same distance offset by a third of the range
/// each, so their median reconstructs the true distance while each one alone
/// still carries usable data at a corner.
fn encode_distance(d: f32) -> [u8; 3] {
    let scaled = (d * 0.5 + 0.5).clamp(0.0, 1.0);
    let v = (scaled * 255.0).round() as u8;
    [v, v, v]
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A hand-built atlas, so the layout tests need no font file.
    fn atlas() -> MsdfAtlas {
        let entries: Vec<GlyphEntry> = "ab"
            .chars()
            .map(|c| GlyphEntry {
                character: c,
                rect: AtlasRect {
                    x: 0,
                    y: 0,
                    width: 8,
                    height: 8,
                },
                advance: 0.5,
                bearing: Vec2::ZERO,
                distance_range: EM_DISTANCE_RANGE,
            })
            .collect();
        MsdfAtlas {
            pixels: vec![0; 16 * 16 * 3],
            width: 16,
            height: 16,
            glyphs: entries,
            px_per_em: 16.0,
            units_per_em: 1000.0,
        }
    }

    #[test]
    fn an_atlas_has_three_channels() {
        assert_eq!(CHANNELS, 3);
        assert_eq!(atlas().channels(), 3);
    }

    #[test]
    fn a_glyph_is_looked_up_by_character() {
        let a = atlas();
        assert!(a.glyph('a').is_some());
        assert!(a.glyph('z').is_none());
    }

    #[test]
    fn an_advance_scales_with_size() {
        let g = atlas().glyph('a').copied().unwrap();
        assert!((g.advance_px(32.0) - 16.0).abs() < 1e-5);
        assert!((g.advance_px(16.0) - 8.0).abs() < 1e-5);
    }

    #[test]
    fn a_zero_size_glyph_does_not_advance() {
        let g = atlas().glyph('a').copied().unwrap();
        assert_eq!(g.advance_px(0.0), 0.0);
        assert_eq!(g.advance_px(-10.0), 0.0);
    }

    #[test]
    fn laying_out_places_every_glyph() {
        let run = layout(&atlas(), "ab", 16.0);
        assert_eq!(run.glyphs.len(), 2);
        assert_eq!(run.glyphs[0].character, 'a');
        assert_eq!(run.glyphs[1].character, 'b');
    }

    #[test]
    fn laying_out_advances_the_pen() {
        let run = layout(&atlas(), "ab", 16.0);
        assert!(
            (run.glyphs[1].position.x - 8.0).abs() < 1e-5,
            "{:?}",
            run.glyphs[1].position
        );
    }

    #[test]
    fn a_run_measures_its_width() {
        let run = layout(&atlas(), "ab", 16.0);
        assert!((run.width - 16.0).abs() < 1e-5, "{}", run.width);
    }

    #[test]
    fn a_space_advances_without_drawing() {
        let run = layout(&atlas(), "a b", 16.0);
        assert_eq!(run.glyphs.len(), 2, "the space is not drawn");
        let gap = run.glyphs[1].position.x - (run.glyphs[0].position.x + 8.0);
        assert!(gap > 0.0, "but it does advance the pen: gap {gap}");
    }

    #[test]
    fn a_newline_starts_a_new_line() {
        let run = layout(&atlas(), "a\nb", 16.0);
        assert_eq!(run.glyphs.len(), 2);
        assert!(
            run.glyphs[1].position.y > run.glyphs[0].position.y,
            "the second glyph is on the next line: {:?}",
            run.glyphs[1].position
        );
    }

    #[test]
    fn a_run_takes_the_width_of_its_longest_line() {
        let run = layout(&atlas(), "ab\na", 16.0);
        assert!((run.width - 16.0).abs() < 1e-5, "{}", run.width);
    }

    #[test]
    fn an_empty_string_lays_out_to_nothing() {
        let run = layout(&atlas(), "", 16.0);
        assert!(run.glyphs.is_empty());
        assert_eq!(run.width, 0.0);
    }

    #[test]
    fn a_missing_character_is_skipped() {
        let run = layout(&atlas(), "az", 16.0);
        assert_eq!(run.glyphs.len(), 1, "z has no glyph, so it is not drawn");
    }

    #[test]
    fn a_zero_size_does_not_divide_by_zero() {
        let run = layout(&atlas(), "ab", 0.0);
        assert!(run.glyphs.iter().all(|g| g.position.x.is_finite()));
    }

    #[test]
    fn right_to_left_mirrors_the_run() {
        let ltr = layout_with_direction(&atlas(), "ab", 16.0, TextDirection::LeftToRight, 1.2);
        let rtl = layout_with_direction(&atlas(), "ab", 16.0, TextDirection::RightToLeft, 1.2);
        assert!(
            rtl.glyphs[0].position.x > rtl.glyphs[1].position.x,
            "the first glyph is rightmost"
        );
        assert!(ltr.glyphs[1].position.x > ltr.glyphs[0].position.x);
    }

    #[test]
    fn a_right_to_left_run_keeps_its_width() {
        let ltr = layout(&atlas(), "ab", 16.0);
        let rtl = layout_with_direction(&atlas(), "ab", 16.0, TextDirection::RightToLeft, 1.2);
        assert!(
            (ltr.width - rtl.width).abs() < 1e-5,
            "direction does not change metrics"
        );
    }

    #[test]
    fn centring_shifts_the_run() {
        let mut run = layout(&atlas(), "ab", 16.0);
        align_run(&mut run, TextAlign::Center, 32.0);
        assert!(
            (run.glyphs[0].position.x - 8.0).abs() < 1e-5,
            "{:?}",
            run.glyphs[0].position
        );
    }

    #[test]
    fn right_alignment_shifts_the_run_further() {
        let mut run = layout(&atlas(), "ab", 16.0);
        align_run(&mut run, TextAlign::Right, 32.0);
        assert!(
            (run.glyphs[0].position.x - 16.0).abs() < 1e-5,
            "{:?}",
            run.glyphs[0].position
        );
    }

    #[test]
    fn left_alignment_does_not_shift() {
        let mut run = layout(&atlas(), "ab", 16.0);
        align_run(&mut run, TextAlign::Left, 32.0);
        assert!(run.glyphs[0].position.x.abs() < 1e-5);
    }

    #[test]
    fn alignment_does_not_change_the_width() {
        let mut run = layout(&atlas(), "ab", 16.0);
        let before = run.width;
        align_run(&mut run, TextAlign::Center, 32.0);
        assert!((run.width - before).abs() < 1e-5);
    }

    #[test]
    fn a_texel_outside_the_atlas_is_none() {
        let a = atlas();
        assert!(a.texel(0, 0).is_some());
        assert!(a.texel(100, 0).is_none());
    }

    #[test]
    fn a_texel_reads_three_normalised_values() {
        let a = atlas();
        let t = a.texel(0, 0).unwrap();
        assert!(t.iter().all(|v| (0.0..=1.0).contains(v)), "{t:?}");
    }

    #[test]
    fn a_rect_fits_only_inside_the_atlas() {
        let r = AtlasRect {
            x: 8,
            y: 8,
            width: 8,
            height: 8,
        };
        assert!(r.fits(16, 16));
        assert!(!r.fits(15, 16), "the right edge is exclusive");
    }

    #[test]
    fn packing_places_rectangles_without_overlap() {
        let rects = vec![(8u32, 8u32); 4];
        let packed = pack(&rects, 32, 32, 1).unwrap();
        assert_eq!(packed.len(), 4);
        for (i, a) in packed.iter().enumerate() {
            for b in packed.iter().skip(i + 1) {
                let overlap_x = a.x < b.right() && b.x < a.right();
                let overlap_y = a.y < b.bottom() && b.y < a.bottom();
                assert!(!(overlap_x && overlap_y), "{a:?} overlaps {b:?}");
            }
        }
    }

    #[test]
    fn packing_starts_a_new_shelf_when_it_must() {
        let rects = vec![(16u32, 8u32); 3];
        let packed = pack(&rects, 20, 32, 1).unwrap();
        // Two fit across 20 pixels, the third goes below them.
        assert!(packed[2].y > packed[0].y, "{packed:?}");
    }

    #[test]
    fn packing_reports_a_set_that_cannot_fit() {
        let rects = vec![(64u32, 64u32)];
        let err = pack(&rects, 16, 16, 1).unwrap_err();
        assert!(matches!(err, GltfError::Decode { .. }), "{err:?}");
    }

    #[test]
    fn a_distance_to_a_segment_is_measured_correctly() {
        // A horizontal segment from (0,0) to (1,0); a point above its middle is
        // half a unit away.
        let d = distance_to_segment([0.5, 0.5], [0.0, 0.0], [1.0, 0.0]);
        assert!((d - 0.5).abs() < 1e-6, "{d}");
    }

    #[test]
    fn a_distance_beyond_a_segment_clamps_to_its_ends() {
        let d = distance_to_segment([2.0, 0.0], [0.0, 0.0], [1.0, 0.0]);
        assert!((d - 1.0).abs() < 1e-6, "{d}");
    }

    #[test]
    fn a_degenerate_segment_measures_to_its_point() {
        let d = distance_to_segment([1.0, 1.0], [0.0, 0.0], [0.0, 0.0]);
        assert!((d - 2f32.sqrt()).abs() < 1e-6, "{d}");
    }

    #[test]
    fn a_point_inside_a_square_contour_is_inside() {
        let square = vec![vec![
            [-1.0, -1.0],
            [1.0, -1.0],
            [1.0, 1.0],
            [-1.0, 1.0],
            [-1.0, -1.0],
        ]];
        assert!(point_in_contours(&square, [0.0, 0.0]));
    }

    #[test]
    fn a_point_outside_a_square_contour_is_outside() {
        let square = vec![vec![
            [-1.0, -1.0],
            [1.0, -1.0],
            [1.0, 1.0],
            [-1.0, 1.0],
            [-1.0, -1.0],
        ]];
        assert!(!point_in_contours(&square, [2.0, 0.0]));
    }

    #[test]
    fn winding_does_not_change_whether_a_point_is_inside() {
        // The even-odd rule makes the sign independent of which way a contour
        // was wound, which is what lets the distance be signed.
        let ccw = vec![vec![
            [-1.0, -1.0],
            [1.0, -1.0],
            [1.0, 1.0],
            [-1.0, 1.0],
            [-1.0, -1.0],
        ]];
        let cw = vec![vec![
            [-1.0, -1.0],
            [-1.0, 1.0],
            [1.0, 1.0],
            [1.0, -1.0],
            [-1.0, -1.0],
        ]];
        assert_eq!(
            point_in_contours(&ccw, [0.0, 0.0]),
            point_in_contours(&cw, [0.0, 0.0])
        );
    }

    #[test]
    fn a_square_contour_measures_a_centre_distance_of_one() {
        let square = vec![vec![
            [-1.0, -1.0],
            [1.0, -1.0],
            [1.0, 1.0],
            [-1.0, 1.0],
            [-1.0, -1.0],
        ]];
        let d = signed_distance(&square, [0.0, 0.0]);
        assert!(
            (d - 1.0).abs() < 1e-5,
            "the centre is one unit from the edge: {d}"
        );
    }

    #[test]
    fn a_signed_distance_is_positive_inside_and_negative_outside() {
        let square = vec![vec![
            [-1.0, -1.0],
            [1.0, -1.0],
            [1.0, 1.0],
            [-1.0, 1.0],
            [-1.0, -1.0],
        ]];
        let inside = signed_distance(&square, [0.0, 0.0]);
        let outside = signed_distance(&square, [3.0, 0.0]);
        assert!(inside > 0.0, "{inside}");
        assert!(outside < 0.0, "{outside}");
    }

    #[test]
    fn a_signed_distance_is_measured_in_ems_not_pixels() {
        let square = vec![vec![
            [-1.0, -1.0],
            [1.0, -1.0],
            [1.0, 1.0],
            [-1.0, 1.0],
            [-1.0, -1.0],
        ]];
        // The distance is in ems, so it does not scale with the pixel size:
        // the bitmap scales, not the field's units.
        let d = signed_distance(&square, [0.0, 0.0]);
        assert!((d - 1.0).abs() < 1e-5, "{d}");
    }

    #[test]
    fn an_empty_contour_set_is_outside() {
        let d = signed_distance(&[], [0.0, 0.0]);
        assert!(d < 0.0, "{d}");
        assert!(d.is_finite(), "an empty outline must not produce infinity");
    }

    #[test]
    fn a_distance_encodes_to_the_middle_of_the_range() {
        // Zero distance is the glyph edge, which is the middle of the range.
        let e = encode_distance(0.0);
        assert!(e.iter().all(|v| (120..=135).contains(v)), "{e:?}");
    }

    #[test]
    fn an_inside_distance_encodes_above_the_middle() {
        // A distance of half the range is the saturated end of the encoding.
        assert_eq!(
            encode_distance(0.5),
            [191, 191, 191],
            "{:?}",
            encode_distance(0.5)
        );
    }

    #[test]
    fn an_outside_distance_encodes_below_the_middle() {
        assert_eq!(
            encode_distance(-0.5),
            [64, 64, 64],
            "{:?}",
            encode_distance(-0.5)
        );
    }

    #[test]
    fn a_distance_beyond_the_range_saturates() {
        assert_eq!(encode_distance(10.0), [255, 255, 255]);
        assert_eq!(encode_distance(-10.0), [0, 0, 0]);
    }

    #[test]
    fn an_encode_is_monotonic_in_the_distance() {
        // A larger distance must never encode to a smaller byte, or the field
        // would invert halfway through a glyph's interior.
        let mut previous = 0u8;
        for step in 0..=20 {
            let d = -1.0 + step as f32 * 0.1;
            let encoded = encode_distance(d)[0];
            assert!(
                encoded >= previous,
                "not monotonic at {d}: {previous} then {encoded}"
            );
            previous = encoded;
        }
    }

    #[test]
    fn an_encode_uses_the_whole_byte_range() {
        let low = encode_distance(-10.0)[0];
        let high = encode_distance(10.0)[0];
        assert_eq!(low, 0);
        assert_eq!(high, 255);
        assert!(
            high > low,
            "a field that never spans the range cannot be interpolated"
        );
    }

    /// A real font on this machine, or `None` when there is none.
    ///
    /// The tests that use this return early when it is `None`, which is
    /// honest but easy to miss: a hard-coded path that does not exist makes
    /// every one of them pass without running anything.
    fn system_font() -> Option<Vec<u8>> {
        const CANDIDATES: &[&str] = &[
            "/usr/share/fonts/noto/NotoSans-Regular.ttf",
            "/usr/share/fonts/TTF/DejaVuSans.ttf",
            "/usr/share/fonts/gnu-free/FreeSans.ttf",
            "/usr/share/fonts/TTF/LiberationSans-Regular.ttf",
        ];
        CANDIDATES
            .iter()
            .find(|p| std::path::Path::new(p).exists())
            .and_then(|p| std::fs::read(p).ok())
    }

    #[test]
    fn a_real_font_builds_an_atlas() {
        let Some(data) = system_font() else {
            // No system font here. The arithmetic above is still covered, and
            // saying so beats a test that passes without running.
            eprintln!("skipped: no system font on this machine");
            return;
        };
        let font = MsdfFont::new(&data, "ABgjy", 32.0, 4).unwrap();
        assert!(!font.atlas.glyphs.is_empty());
        assert!(font.atlas.pixels.len() > 0);
        assert!(font.atlas.width >= 16);
    }

    #[test]
    fn a_real_font_measures_advances_that_increase() {
        let Some(data) = system_font() else { return };
        let font = MsdfFont::new(&data, "iW", 32.0, 4).unwrap();
        let i = font.atlas.glyph('i').unwrap();
        let w = font.atlas.glyph('W').unwrap();
        assert!(
            w.advance > i.advance,
            "a W is wider than an i: {} vs {}",
            w.advance,
            i.advance
        );
    }

    #[test]
    fn a_real_font_lays_out_a_string_that_advances() {
        let Some(data) = system_font() else { return };
        let font = MsdfFont::new(&data, "hello", 32.0, 4).unwrap();
        let run = layout(&font.atlas, "hello", 32.0);
        assert_eq!(run.glyphs.len(), 5);
        assert!(run.width > 0.0, "a laid-out run has width");
        for pair in run.glyphs.windows(2) {
            assert!(
                pair[1].position.x > pair[0].position.x,
                "each glyph is placed after the last"
            );
        }
    }

    #[test]
    fn garbage_font_bytes_are_a_decode_error() {
        let err = MsdfFont::new(b"not a font", "a", 16.0, 4).unwrap_err();
        assert!(matches!(err, GltfError::Decode { .. }), "{err:?}");
    }
}
