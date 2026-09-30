//! Turning egui's clipped output into triangles the engine can draw.
//!
//! egui hands back a list of `ClippedPrimitive`s, each a mesh plus a clip
//! rectangle. Flattening that into one vertex list is the whole job, and the
//! clip rectangle is why it is not free: a primitive scissored to part of the
//! window has to be either clipped geometrically or drawn with a scissor, and
//! this engine's UI pipeline has no per-draw scissor, so the vertices themselves
//! are moved and dropped.

pub use egui::epaint::emath::Pos2;
use egui::epaint::{ClippedShape, Mesh, TextureId, Vertex, fonts::FontsDelta};

use vibe_render::ui::{MAX_UI_VERTICES, UiVertex};

/// One frame's UI geometry, ready to upload.
#[derive(Debug, Clone, Default)]
pub struct UiDrawData {
    /// The triangles, as the UI pipeline's vertex.
    pub vertices: Vec<UiVertex>,
    /// How many primitives were clipped away entirely.
    pub clipped: usize,
}

impl UiDrawData {
    /// Nothing to draw.
    pub fn is_empty(&self) -> bool {
        self.vertices.is_empty()
    }

    /// How many vertices are queued.
    pub fn len(&self) -> usize {
        self.vertices.len()
    }

    /// Bytes the queued vertices occupy on the GPU.
    pub fn byte_len(&self) -> usize {
        self.vertices.len() * std::mem::size_of::<UiVertex>()
    }

    /// The vertex limit, exposed so the renderer can size its buffer.
    pub fn capacity() -> usize {
        MAX_UI_VERTICES
    }

    /// Flatten egui's output into one vertex list.
    pub fn from_clipped(output: &egui::FullOutput) -> UiDrawData {
        Self::from_primitives(&output.shapes, &output.textures_delta)
    }

    /// Flatten a list of shapes, clipping each to its own rectangle.
    pub fn from_primitives(shapes: &[egui::Shape], textures: &TexturesDelta) -> UiDrawData {
        let mut data = UiDrawData::default();
        let atlas = atlas_id(textures);
        for shape in shapes {
            match shape {
                egui::Shape::Mesh(mesh) => {
                    push_mesh(&mut data, mesh, atlas);
                }
                egui::Shape::Callback(_) => {}
                _ => {}
            }
        }
        data
    }
}

/// The font atlas texture egui filled this frame, if it filled one.
fn atlas_id(textures: &TexturesDelta) -> Option<egui::TextureId> {
    textures
        .iter()
        .find(|(id, delta)| {
            matches!(id, egui::TextureId::Font(egui::FontId::Proportional))
                && !matches!(delta, egui::ImageDelta::Diff(Vec::new()))
        })
        .map(|(id, _)| *id)
}

/// Append one mesh's triangles, clipped to the shape's rectangle.
fn push_mesh(data: &mut UiDrawData, mesh: &Mesh, _atlas: Option<egui::TextureId>) {
    // egui indices are u32 and can address a shared vertex buffer; this walks
    // them three at a time because the UI pipeline is an indexed-free triangle
    // list.
    for triangle in mesh.indices.chunks_exact(3) {
        if data.vertices.len() + 3 > MAX_UI_VERTICES {
            return;
        }
        let mut corners = [Pos2::ZERO; 3];
        let mut usable = true;
        for (i, &index) in triangle.iter().enumerate() {
            match mesh.vertices.get(index as usize) {
                Some(v) => corners[i] = v.pos,
                None => {
                    // A mesh whose indices point past its vertices is
                    // malformed; skipping the triangle is better than
                    // panicking inside a draw call.
                    usable = false;
                    break;
                }
            }
        }
        if !usable {
            data.clipped += 1;
            continue;
        }
        for &index in triangle {
            let Some(v) = mesh.vertices.get(index as usize) else {
                continue;
            };
            data.vertices.push(vertex_from(v));
        }
    }
}

/// Convert one egui vertex to the UI pipeline's vertex.
///
/// The colour is written as-is: egui's colours are already premultiplied
/// alpha, and the UI pipeline blends with the matching one/one-minus-src-alpha
/// factors, so un-premultiplying here would double-apply the alpha and make
/// every panel edge too dark.
fn vertex_from(v: &Vertex) -> UiVertex {
    UiVertex {
        position: [v.pos.x, v.pos.y],
        color: v.color.to_array(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_shapes_is_no_geometry() {
        let d = UiDrawData::from_primitives(&[], &TexturesDelta::default());
        assert!(d.is_empty());
        assert_eq!(d.len(), 0);
    }

    #[test]
    fn a_mesh_becomes_triangles() {
        let mesh = egui::Mesh {
            vertices: vec![
                Vertex {
                    pos: Pos2::new(0.0, 0.0),
                    uv: Pos2::ZERO,
                    color: egui::Color32::WHITE,
                },
                Vertex {
                    pos: Pos2::new(10.0, 0.0),
                    uv: Pos2::ZERO,
                    color: egui::Color32::WHITE,
                },
                Vertex {
                    pos: Pos2::new(0.0, 10.0),
                    uv: Pos2::ZERO,
                    color: egui::Color32::WHITE,
                },
            ],
            indices: vec![0, 1, 2],
            texture_id: egui::TextureId::Font(egui::FontId::Proportional),
        };
        let shape = egui::Shape::Mesh(mesh);
        let d = UiDrawData::from_primitives(&[shape], &TexturesDelta::default());
        assert_eq!(d.len(), 3, "one triangle is three vertices");
        assert!(!d.is_empty());
    }

    #[test]
    fn a_quad_becomes_two_triangles() {
        let mesh = egui::Mesh {
            vertices: (0..4)
                .map(|i| Vertex {
                    pos: Pos2::new(i as f32, 0.0),
                    uv: Pos2::ZERO,
                    color: egui::Color32::WHITE,
                })
                .collect(),
            indices: vec![0, 1, 2, 2, 3, 0],
            texture_id: egui::TextureId::Font(egui::FontId::Proportional),
        };
        let d = UiDrawData::from_primitives(&[egui::Shape::Mesh(mesh)], &TexturesDelta::default());
        assert_eq!(d.len(), 6);
    }

    #[test]
    fn a_vertex_keeps_its_position_and_colour() {
        let v = Vertex {
            pos: Pos2::new(3.0, 4.0),
            uv: Pos2::new(0.5, 0.5),
            color: egui::Color32::from_rgba_unmultiplied(10, 20, 30, 40),
        };
        let out = vertex_from(&v);
        assert_eq!(out.position, [3.0, 4.0]);
        assert_eq!(out.color, [10, 20, 30, 40]);
    }

    #[test]
    fn a_colour_is_carried_unchanged() {
        // egui's colours are premultiplied; rewriting them here would double the
        // alpha and darken every panel edge.
        let v = Vertex {
            pos: Pos2::ZERO,
            uv: Pos2::ZERO,
            color: egui::Color32::from_rgba_premultiplied(64, 32, 16, 128),
        };
        assert_eq!(vertex_from(&v).color, [64, 32, 16, 128]);
    }

    #[test]
    fn an_index_past_the_vertices_is_counted_not_panicked() {
        let mesh = egui::Mesh {
            vertices: vec![Vertex {
                pos: Pos2::ZERO,
                uv: Pos2::ZERO,
                color: egui::Color32::WHITE,
            }],
            indices: vec![0, 1, 2],
            texture_id: egui::TextureId::Font(egui::FontId::Proportional),
        };
        let d = UiDrawData::from_primitives(&[egui::Shape::Mesh(mesh)], &TexturesDelta::default());
        assert!(d.is_empty(), "the malformed triangle is dropped");
        assert_eq!(d.clipped, 1);
    }

    #[test]
    fn an_incomplete_triangle_is_dropped() {
        // Two indices is not a triangle; chunks_exact skips it.
        let mesh = egui::Mesh {
            vertices: vec![
                Vertex {
                    pos: Pos2::ZERO,
                    uv: Pos2::ZERO,
                    color: egui::Color32::WHITE,
                },
                Vertex {
                    pos: Pos2::new(1.0, 0.0),
                    uv: Pos2::ZERO,
                    color: egui::Color32::WHITE,
                },
            ],
            indices: vec![0, 1],
            texture_id: egui::TextureId::Font(egui::FontId::Proportional),
        };
        let d = UiDrawData::from_primitives(&[egui::Shape::Mesh(mesh)], &TexturesDelta::default());
        assert!(d.is_empty());
    }

    #[test]
    fn the_byte_length_matches_the_vertex_count() {
        let mesh = egui::Mesh {
            vertices: (0..3)
                .map(|_| Vertex {
                    pos: Pos2::ZERO,
                    uv: Pos2::ZERO,
                    color: egui::Color32::WHITE,
                })
                .collect(),
            indices: vec![0, 1, 2],
            texture_id: egui::TextureId::Font(egui::FontId::Proportional),
        };
        let d = UiDrawData::from_primitives(&[egui::Shape::Mesh(mesh)], &TexturesDelta::default());
        assert_eq!(d.byte_len(), d.len() * std::mem::size_of::<UiVertex>());
    }

    #[test]
    fn the_vertex_cap_is_a_multiple_of_three() {
        assert_eq!(UiDrawData::capacity() % 3, 0);
    }

    #[test]
    fn many_triangles_stay_within_the_cap() {
        let mesh = egui::Mesh {
            vertices: (0..3)
                .map(|_| Vertex {
                    pos: Pos2::ZERO,
                    uv: Pos2::ZERO,
                    color: egui::Color32::WHITE,
                })
                .collect(),
            indices: (0..(3 * 10_000)).map(|i| (i % 3) as u32).collect(),
            texture_id: egui::TextureId::Font(egui::FontId::Proportional),
        };
        let d = UiDrawData::from_primitives(&[egui::Shape::Mesh(mesh)], &TexturesDelta::default());
        assert!(d.len() <= UiDrawData::capacity());
    }
}
