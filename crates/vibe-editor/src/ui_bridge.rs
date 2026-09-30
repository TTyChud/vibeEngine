//! Turning egui's clipped output into triangles the engine can draw.
//!
//! egui hands back a list of `ClippedShape`s: each a mesh plus the rectangle it
//! was drawn into. Both halves matter. The mesh is already a triangle list, so
//! the conversion is mechanical; the clip rectangle is what keeps a panel's
//! contents inside the panel, and dropping it draws a scrolled list straight
//! over the viewport behind it.
//!
//! The clip is kept as data rather than applied to the vertices. Clipping
//! triangles in software is fiddly and lossy — a clipped quad becomes a polygon
//! that has to be re-triangulated — whereas the GPU already has a scissor
//! rectangle, and the UI pipeline declares one dynamic precisely for this.

use egui::epaint::{ClippedPrimitive, Mesh, Primitive, Vertex};

use vibe_render::ui::{MAX_UI_VERTICES, UiVertex};

/// One mesh's triangles and the rectangle they are clipped to.
#[derive(Debug, Clone, Default)]
pub struct UiBatch {
    /// The triangles, as the UI pipeline's vertex.
    pub vertices: Vec<UiVertex>,
    /// The clip rectangle as `(min_x, min_y, max_x, max_y)`, in pixels.
    pub clip: (f32, f32, f32, f32),
}

impl UiBatch {
    /// True when the batch has nothing to draw.
    pub fn is_empty(&self) -> bool {
        self.vertices.is_empty()
    }
}

/// The font atlas egui rasterised this frame.
///
/// A coverage mask: the red channel of each texel is the glyph's alpha. Stored
/// as RGBA8 because that is the format the engine's other textures use and
/// `R8_UNORM` needs a feature this device does not report.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct FontAtlas {
    /// The atlas width in pixels.
    pub width: u32,
    /// The atlas height in pixels.
    pub height: u32,
    /// RGBA8 texels, row by row from the top.
    pub pixels: Vec<u8>,
}

impl FontAtlas {
    /// True when there is nothing to upload.
    pub fn is_empty(&self) -> bool {
        self.pixels.is_empty() || self.width == 0 || self.height == 0
    }

    /// Bytes a row occupies, which is four per texel.
    pub fn stride(&self) -> usize {
        self.width as usize * 4
    }

    /// Bytes a full upload of this atlas needs.
    pub fn byte_len(&self) -> usize {
        self.stride() * self.height as usize
    }
}

/// One frame's UI geometry, ready to upload.
#[derive(Debug, Clone, Default)]
pub struct UiDrawData {
    /// The batches, in draw order.
    pub batches: Vec<UiBatch>,
    /// How many shapes were skipped entirely.
    pub clipped: usize,
    /// How many triangles were skipped for exceeding the vertex cap.
    pub dropped: usize,
    /// The font atlas, when egui rasterised one this frame.
    ///
    /// Empty on every frame after the first, because egui rasterises the atlas
    /// once and reuses it. A renderer that uploads whenever this is non-empty
    /// therefore uploads once — which is also why the atlas has to be kept by
    /// the caller rather than taken from here: the frame that carries it is
    /// whichever one egui rasterised on, and a frame that runs before the GPU
    /// exists would otherwise lose it with nothing to replace it.
    pub atlas: FontAtlas,
}

impl UiDrawData {
    /// The font atlas egui rasterised, if it rasterised one this frame.
    ///
    /// egui's font texture is a single-channel coverage mask delivered as an
    /// RGBA image, so the red channel is the glyph alpha and the other three
    /// are unused. It is reduced to one byte per texel here because that is what
    /// an `R8_UNORM` upload wants, and uploading RGBA would be four times the
    /// bandwidth for a mask the shader only ever reads one channel of.
    pub fn font_atlas(output: &egui::FullOutput) -> FontAtlas {
        // Managed(0) is the font texture: epaint's first managed slot is
        // where the atlas lives, and it is the default so `Managed(0)` is
        // exactly right rather than a guess.
        let id = egui::TextureId::Managed(0);
        let Some(deltas) = output.textures_delta.set.get(&id) else {
            return FontAtlas::default();
        };
        // A patch rather than a whole image would need the previous contents to
        // patch into, which a backend that keeps its own atlas copy has and one
        // that does not does not; the whole image is the case that matters.
        let Some(delta) = deltas.last() else {
            return FontAtlas::default();
        };
        if delta.pos.is_some() {
            return FontAtlas::default();
        }
        // One variant, so this always binds; a second would arrive as a new
        // enum variant and this would need to handle it.
        let egui::epaint::ImageData::Color(image) = &delta.image;
        let [w, h] = image.size;
        if w == 0 || h == 0 {
            return FontAtlas::default();
        }
        let raw = image.as_raw();
        let mut pixels = Vec::with_capacity(w * h * 4);
        for texel in raw.chunks_exact(4) {
            // The coverage goes in every channel rather than only red. R8_UNORM
            // as a sampled image needs a feature this device does not report,
            // and creating one crashes the driver; RGBA8 is what the engine
            // already samples, and the shader reads red either way. The cost is
            // four bytes per texel of an atlas that is written once.
            pixels.extend_from_slice(texel);
        }
        FontAtlas {
            width: w as u32,
            height: h as u32,
            pixels,
        }
    }

    /// Nothing to draw.
    pub fn is_empty(&self) -> bool {
        self.batches.iter().all(|b| b.is_empty())
    }

    /// How many vertices are queued across every batch.
    pub fn len(&self) -> usize {
        self.batches.iter().map(|b| b.vertices.len()).sum()
    }

    /// How many batches are queued.
    pub fn batch_count(&self) -> usize {
        self.batches.len()
    }

    /// Bytes the queued vertices occupy on the GPU.
    pub fn byte_len(&self) -> usize {
        self.len() * std::mem::size_of::<UiVertex>()
    }

    /// The vertex cap, exposed so the renderer can size its buffer.
    pub fn capacity() -> usize {
        MAX_UI_VERTICES
    }

    /// Tessellate egui's output and flatten it into batches.
    ///
    /// The tessellation step is not optional. `FullOutput::shapes` holds the
    /// *description* of what to paint — rectangles, text runs, paths — and only
    /// `Context::tessellate` turns those into triangles. Reading `shapes`
    /// directly and looking for meshes finds nothing, because a frame of egui
    /// emits almost no `Shape::Mesh` at all: the shapes are rectangles and
    /// glyphs, and the meshes come out the far end of the tessellator.
    pub fn from_output(context: &egui::Context, output: &egui::FullOutput) -> UiDrawData {
        let primitives = context.tessellate(output.shapes.clone(), output.pixels_per_point);
        Self::from_primitives(&primitives)
    }

    /// Flatten tessellated primitives into batches.
    ///
    /// Consecutive primitives under one clip rectangle share a batch: a panel is
    /// usually several primitives under a single scissor, and one vertex range
    /// per panel rather than per primitive is what keeps the draw count at "one
    /// per visible panel" instead of "one per widget".
    pub fn from_primitives(primitives: &[ClippedPrimitive]) -> UiDrawData {
        let mut data = UiDrawData::default();
        let mut budget = MAX_UI_VERTICES;
        let mut current: Option<UiBatch> = None;

        for primitive in primitives {
            let Primitive::Mesh(mesh) = &primitive.primitive else {
                continue;
            };
            let clip = clip_bounds(primitive);
            if !clip_is_visible(clip) {
                data.clipped += 1;
                continue;
            }
            let mut batch = UiBatch {
                vertices: Vec::new(),
                clip,
            };
            let (added, skipped) = append_mesh(&mut batch, mesh, budget);
            if added == 0 {
                data.dropped += skipped;
                continue;
            }
            budget -= added;
            data.dropped += skipped;
            match &mut current {
                Some(open) if same_clip(open.clip, clip) => {
                    open.vertices.extend(batch.vertices);
                }
                Some(open) => {
                    data.batches.push(std::mem::replace(open, batch));
                }
                None => current = Some(batch),
            }
        }
        if let Some(batch) = current
            && !batch.is_empty()
        {
            data.batches.push(batch);
        }
        data
    }
}

/// The clip rectangle of a primitive, as `(min_x, min_y, max_x, max_y)`.
fn clip_bounds(primitive: &ClippedPrimitive) -> (f32, f32, f32, f32) {
    let r = primitive.clip_rect;
    (r.min.x, r.min.y, r.max.x, r.max.y)
}

/// True when a clip rectangle could contribute a visible pixel.
///
/// A non-finite rectangle is treated as visible: it is a malformed frame, and
/// dropping it would blank the editor for as long as the malformation lasts,
/// which is a far worse failure than drawing a panel in the wrong place for one
/// frame.
fn clip_is_visible(clip: (f32, f32, f32, f32)) -> bool {
    let (min_x, min_y, max_x, max_y) = clip;
    if !min_x.is_finite() || !min_y.is_finite() || !max_x.is_finite() || !max_y.is_finite() {
        return true;
    }
    max_x > 0.0 && max_y > 0.0 && min_x < 1.0e5 && min_y < 1.0e5
}

/// True when two clip rectangles would produce the same scissor.
///
/// Compared after rounding, because the scissor is integral: two rectangles a
/// hundredth of a pixel apart clip identically, so treating them as different
/// would split a batch for nothing.
fn same_clip(a: (f32, f32, f32, f32), b: (f32, f32, f32, f32)) -> bool {
    let r = |v: f32| {
        let v = if v.is_finite() { v } else { 0.0 };
        (v * 64.0).round() / 64.0
    };
    (r(a.0), r(a.1), r(a.2), r(a.3)) == (r(b.0), r(b.1), r(b.2), r(b.3))
}

/// Append one mesh's triangles under a clip rectangle, within a budget.
///
/// Indices are walked three at a time because the UI pipeline reads a flat
/// triangle stream rather than an index buffer: egui's meshes share vertices,
/// and the pipeline has no index buffer to share them through.
fn append_mesh(batch: &mut UiBatch, mesh: &Mesh, budget: usize) -> (usize, usize) {
    let mut added = 0usize;
    let mut skipped = 0usize;
    for triangle in mesh.indices.chunks_exact(3) {
        if added + 3 > budget {
            skipped += 1;
            continue;
        }
        let mut usable = true;
        for &index in triangle {
            if mesh.vertices.get(index as usize).is_none() {
                // A mesh whose indices point past its vertices is malformed;
                // skipping the triangle beats panicking inside a draw call.
                usable = false;
                break;
            }
        }
        if !usable {
            skipped += 1;
            continue;
        }
        for &index in triangle {
            if let Some(v) = mesh.vertices.get(index as usize) {
                batch.vertices.push(vertex_from(v));
            }
        }
        added += 3;
    }
    (added, skipped)
}

/// Convert one egui vertex to the UI pipeline's vertex.
///
/// The colour is written as-is: egui's colours are already premultiplied
/// alpha, and the UI pipeline blends with the matching one/one-minus-src-alpha
/// factors, so un-premultiplying here would apply the alpha twice and make every
/// panel edge too dark.
fn vertex_from(v: &Vertex) -> UiVertex {
    UiVertex {
        position: [v.pos.x, v.pos.y],
        color: v.color.to_array(),
        // The glyph's coordinate in the font atlas. Without it the fragment
        // stage has nothing to sample and every character draws as a solid quad.
        uv: [v.uv.x, v.uv.y],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use egui::epaint::{
        TextureId,
        emath::{Pos2, Rect},
    };

    fn vertex(x: f32, y: f32) -> Vertex {
        Vertex {
            pos: Pos2::new(x, y),
            uv: Pos2::ZERO,
            color: egui::Color32::WHITE,
        }
    }

    fn mesh_of(indices: Vec<u32>, vertices: Vec<Vertex>) -> Mesh {
        Mesh {
            indices,
            vertices,
            texture_id: TextureId::default(),
        }
    }

    fn clipped(clip: Rect, mesh: Mesh) -> ClippedPrimitive {
        ClippedPrimitive {
            clip_rect: clip,
            primitive: Primitive::Mesh(mesh),
        }
    }

    fn triangle() -> Mesh {
        mesh_of(
            vec![0, 1, 2],
            vec![vertex(0.0, 0.0), vertex(10.0, 0.0), vertex(0.0, 10.0)],
        )
    }

    fn full_screen() -> Rect {
        Rect::from_min_max(Pos2::new(0.0, 0.0), Pos2::new(1920.0, 1080.0))
    }

    #[test]
    fn no_shapes_is_no_geometry() {
        let d = UiDrawData::from_primitives(&[]);
        assert!(d.is_empty());
        assert_eq!(d.len(), 0);
    }

    #[test]
    fn a_mesh_becomes_triangles() {
        let d = UiDrawData::from_primitives(&[clipped(full_screen(), triangle())]);
        assert_eq!(d.len(), 3);
        assert!(!d.is_empty());
    }

    #[test]
    fn a_quad_becomes_six_vertices() {
        let mesh = mesh_of(
            vec![0, 1, 2, 2, 3, 0],
            (0..4).map(|i| vertex(i as f32, 0.0)).collect(),
        );
        assert_eq!(
            UiDrawData::from_primitives(&[clipped(full_screen(), mesh)]).len(),
            6
        );
    }

    #[test]
    fn meshes_under_one_clip_share_a_batch() {
        let d = UiDrawData::from_primitives(&[
            clipped(full_screen(), triangle()),
            clipped(full_screen(), triangle()),
        ]);
        assert_eq!(d.batch_count(), 1, "one panel is one draw call");
        assert_eq!(d.len(), 6);
    }

    #[test]
    fn different_clips_make_different_batches() {
        let other = Rect::from_min_max(Pos2::new(0.0, 0.0), Pos2::new(400.0, 1080.0));
        let d = UiDrawData::from_primitives(&[
            clipped(full_screen(), triangle()),
            clipped(other, triangle()),
        ]);
        assert_eq!(d.batch_count(), 2);
    }

    #[test]
    fn a_batch_carries_its_clip() {
        let other = Rect::from_min_max(Pos2::new(10.0, 20.0), Pos2::new(400.0, 1080.0));
        let d = UiDrawData::from_primitives(&[clipped(other, triangle())]);
        assert_eq!(d.batches[0].clip, (10.0, 20.0, 400.0, 1080.0));
    }

    #[test]
    fn a_vertex_keeps_its_position_and_colour() {
        let v = Vertex {
            pos: Pos2::new(3.0, 4.0),
            uv: Pos2::new(0.5, 0.5),
            color: egui::Color32::from_rgba_premultiplied(10, 20, 30, 40),
        };
        let out = vertex_from(&v);
        assert_eq!(out.position, [3.0, 4.0]);
        assert_eq!(out.color, [10, 20, 30, 40]);
    }

    #[test]
    fn a_vertex_keeps_its_texture_coordinate() {
        let v = Vertex {
            pos: Pos2::new(1.0, 2.0),
            uv: Pos2::new(0.25, 0.75),
            color: egui::Color32::WHITE,
        };
        assert_eq!(vertex_from(&v).uv, [0.25, 0.75]);
    }

    #[test]
    fn a_premultiplied_colour_is_carried_unchanged() {
        let v = Vertex {
            pos: Pos2::ZERO,
            uv: Pos2::ZERO,
            color: egui::Color32::from_rgba_premultiplied(64, 32, 16, 128),
        };
        assert_eq!(vertex_from(&v).color, [64, 32, 16, 128]);
    }

    #[test]
    fn an_index_past_the_vertices_is_skipped() {
        let mesh = mesh_of(vec![0, 1, 2], vec![vertex(0.0, 0.0)]);
        let mut batch = UiBatch::default();
        let (added, skipped) = append_mesh(&mut batch, &mesh, MAX_UI_VERTICES);
        assert_eq!(added, 0);
        assert_eq!(skipped, 1, "the malformed triangle is counted");
    }

    #[test]
    fn an_incomplete_triangle_is_skipped() {
        let mesh = mesh_of(vec![0, 1], vec![vertex(0.0, 0.0), vertex(1.0, 0.0)]);
        let mut batch = UiBatch::default();
        let (added, _) = append_mesh(&mut batch, &mesh, MAX_UI_VERTICES);
        assert_eq!(added, 0, "two indices are not a triangle");
    }

    #[test]
    fn a_clip_entirely_off_screen_is_dropped() {
        let off = Rect::from_min_max(Pos2::new(-5000.0, -5000.0), Pos2::new(-4000.0, -4000.0));
        let d = UiDrawData::from_primitives(&[clipped(off, triangle())]);
        assert!(d.is_empty());
        assert_eq!(d.clipped, 1);
    }

    #[test]
    fn a_non_finite_clip_is_kept_rather_than_dropped() {
        let bad = Rect::from_min_max(Pos2::new(f32::NAN, 0.0), Pos2::new(100.0, 100.0));
        let d = UiDrawData::from_primitives(&[clipped(bad, triangle())]);
        assert!(!d.is_empty(), "a malformed frame must not blank the editor");
    }

    #[test]
    fn the_byte_length_matches_the_vertex_count() {
        let d = UiDrawData::from_primitives(&[clipped(full_screen(), triangle())]);
        assert_eq!(d.byte_len(), d.len() * std::mem::size_of::<UiVertex>());
    }

    #[test]
    fn the_vertex_cap_is_a_multiple_of_three() {
        // A triangle list cannot draw a partial triangle, so a cap that is not a
        // multiple of three would leave one at the end.
        assert_eq!(UiDrawData::capacity() % 3, 0);
    }

    #[test]
    fn a_mesh_beyond_the_budget_is_truncated_not_unbounded() {
        let mesh = mesh_of(
            (0..3 * 10).map(|i| (i % 3) as u32).collect(),
            (0..3).map(|i| vertex(i as f32, 0.0)).collect(),
        );
        let mut batch = UiBatch::default();
        let (added, skipped) = append_mesh(&mut batch, &mesh, 6);
        assert_eq!(added, 6, "only the budget is taken");
        assert_eq!(skipped, 8, "the other eight triangles are dropped");
        assert_eq!(batch.vertices.len(), 6);
    }

    #[test]
    fn an_empty_batch_reports_itself() {
        assert!(UiBatch::default().is_empty());
    }
}
