//! The quad batcher: many quads, one draw call.
//!
//! Quads are appended to a CPU-side vertex buffer in a single allocation. When
//! the buffer would overflow, the batch is flushed rather than growing without
//! bound, so a frame that tries to draw a million quads costs a few draw calls
//! instead of an allocation.

use vibe_math::Aabb2;
use vibe_rhi::{BatchStats, QuadVertex, SubTexture};

use crate::error::RenderError;

/// How many quads one buffer holds.
///
/// Six vertices per quad, 32 bytes each: a 64k-quad batch is 12 MiB, which is
/// far more than a 2D game needs and small enough to stay resident.
pub const DEFAULT_BATCH_CAPACITY: usize = 65_536;

/// Collects quads into one vertex buffer and reports what the frame cost.
#[derive(Debug)]
pub struct Batcher {
    vertices: Vec<QuadVertex>,
    capacity: usize,
    /// Number of batches flushed this frame.
    pub batches: usize,
    /// Quads accepted this frame.
    pub quads: usize,
    /// Quads dropped because the buffer was full.
    pub dropped: usize,
}

impl Default for Batcher {
    fn default() -> Self {
        Batcher::new(DEFAULT_BATCH_CAPACITY)
    }
}

impl Batcher {
    /// A batcher holding at most `capacity` quads.
    pub fn new(capacity: usize) -> Batcher {
        Batcher {
            vertices: Vec::with_capacity(capacity * 6),
            capacity,
            batches: 0,
            quads: 0,
            dropped: 0,
        }
    }

    /// How many quads the buffer holds.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// How many quads are queued.
    pub fn len(&self) -> usize {
        self.quads
    }

    /// True when nothing is queued.
    pub fn is_empty(&self) -> bool {
        self.quads == 0
    }

    /// Empty the batcher for a new frame, keeping the allocation.
    pub fn clear(&mut self) {
        self.vertices.clear();
        self.batches = 0;
        self.quads = 0;
        self.dropped = 0;
    }

    /// The queued vertices.
    pub fn vertices(&self) -> &[QuadVertex] {
        &self.vertices
    }

    /// How many vertices are queued.
    pub fn vertex_count(&self) -> usize {
        self.vertices.len()
    }

    /// Bytes the queued vertices occupy on the GPU.
    pub fn byte_len(&self) -> usize {
        self.vertices.len() * std::mem::size_of::<QuadVertex>()
    }

    /// Add an axis-aligned quad.
    ///
    /// # Errors
    ///
    /// Returns [`RenderError::BatchTooLarge`] when the buffer is full, so the
    /// caller can flush and retry rather than silently losing the quad.
    pub fn push_quad(
        &mut self,
        position: glam::Vec2,
        size: glam::Vec2,
        color: [u8; 4],
        texture: &SubTexture,
    ) -> Result<usize, RenderError> {
        self.push_quads(position, size, color, texture, 1)
    }

    /// Add several copies of one quad, which is what an explosion or a
    /// particle burst wants.
    ///
    /// # Errors
    ///
    /// Returns [`RenderError::BatchTooLarge`] if the whole run does not fit.
    pub fn push_quads(
        &mut self,
        position: glam::Vec2,
        size: glam::Vec2,
        color: [u8; 4],
        texture: &SubTexture,
        count: usize,
    ) -> Result<usize, RenderError> {
        if self.quads + count > self.capacity {
            return Err(RenderError::BatchTooLarge {
                quads: self.quads + count,
                capacity: self.capacity,
            });
        }

        let first_vertex = self.vertices.len();
        for _ in 0..count {
            // Two triangles per quad: top-left, top-right, bottom-right, and
            // top-left, bottom-right, bottom-left.
            let corners = [
                (position, glam::Vec2::new(0.0, 0.0)),
                (
                    position + glam::Vec2::new(size.x, 0.0),
                    glam::Vec2::new(1.0, 0.0),
                ),
                (position + size, glam::Vec2::new(1.0, 1.0)),
                (position, glam::Vec2::new(0.0, 0.0)),
                (position + size, glam::Vec2::new(1.0, 1.0)),
                (
                    position + glam::Vec2::new(0.0, size.y),
                    glam::Vec2::new(0.0, 1.0),
                ),
            ];
            for (at, uv) in corners {
                self.vertices.push(QuadVertex {
                    position: at,
                    color,
                    uv: texture.uv_origin + uv * texture.uv_size,
                    texture_index: texture.texture_index,
                    _pad: [0; 2],
                });
            }
        }
        self.quads += count;
        Ok(first_vertex)
    }

    /// Add a rotated quad, rotating about its top-left corner.
    ///
    /// # Errors
    ///
    /// Returns [`RenderError::BatchTooLarge`] when the buffer is full.
    pub fn push_rotated(
        &mut self,
        origin: glam::Vec2,
        size: glam::Vec2,
        rotation: f32,
        color: [u8; 4],
        texture: &SubTexture,
    ) -> Result<usize, RenderError> {
        if self.quads + 1 > self.capacity {
            return Err(RenderError::BatchTooLarge {
                quads: self.quads + 1,
                capacity: self.capacity,
            });
        }

        let (s, c) = rotation.sin_cos();
        let rotate = |v: glam::Vec2| glam::Vec2::new(v.x * c - v.y * s, v.x * s + v.y * c);
        let w = rotate(glam::Vec2::new(size.x, 0.0));
        let h = rotate(glam::Vec2::new(0.0, size.y));

        let first_vertex = self.vertices.len();
        let corners = [
            (origin, glam::Vec2::new(0.0, 0.0)),
            (origin + w, glam::Vec2::new(1.0, 0.0)),
            (origin + w + h, glam::Vec2::new(1.0, 1.0)),
            (origin, glam::Vec2::new(0.0, 0.0)),
            (origin + w + h, glam::Vec2::new(1.0, 1.0)),
            (origin + h, glam::Vec2::new(0.0, 1.0)),
        ];
        for (at, uv) in corners {
            self.vertices.push(QuadVertex {
                position: at,
                color,
                uv: texture.uv_origin + uv * texture.uv_size,
                texture_index: texture.texture_index,
                _pad: [0; 2],
            });
        }
        self.quads += 1;
        Ok(first_vertex)
    }

    /// Mark the current contents as a flushed batch.
    pub fn mark_flushed(&mut self) {
        self.batches += 1;
    }

    /// What the frame cost.
    pub fn stats(&self) -> BatchStats {
        BatchStats {
            quads: self.quads as u32,
            draw_calls: self.batches as u32,
            culled: 0,
            vertices: self.vertex_count() as u32,
        }
    }
}

/// Cull a quad against a visible region, so off-screen sprites cost nothing.
///
/// # Safety-free: this is pure arithmetic on the CPU.
pub fn is_quad_visible(bounds: Aabb2, visible: Aabb2) -> bool {
    bounds.min.x < visible.max.x
        && bounds.max.x > visible.min.x
        && bounds.min.y < visible.max.y
        && bounds.max.y > visible.min.y
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::Vec2;

    fn white() -> [u8; 4] {
        [255, 255, 255, 255]
    }

    #[test]
    fn a_new_batcher_is_empty() {
        let b = Batcher::new(16);
        assert!(b.is_empty());
        assert_eq!(b.len(), 0);
        assert_eq!(b.capacity(), 16);
    }

    #[test]
    fn one_quad_is_six_vertices() {
        let mut b = Batcher::new(16);
        b.push_quad(Vec2::ZERO, Vec2::splat(10.0), white(), &SubTexture::full(0))
            .unwrap();
        assert_eq!(b.vertex_count(), 6);
        assert_eq!(b.len(), 1);
    }

    #[test]
    fn a_quad_winds_clockwise_from_the_top_left() {
        let mut b = Batcher::new(4);
        b.push_quad(
            Vec2::ZERO,
            Vec2::new(10.0, 20.0),
            white(),
            &SubTexture::full(0),
        )
        .unwrap();
        let v = b.vertices();
        assert_eq!(v[0].position, Vec2::new(0.0, 0.0));
        assert_eq!(v[1].position, Vec2::new(10.0, 0.0));
        assert_eq!(v[2].position, Vec2::new(10.0, 20.0));
        assert_eq!(v[5].position, Vec2::new(0.0, 20.0));
    }

    #[test]
    fn uvs_cover_the_whole_sub_texture() {
        let mut b = Batcher::new(4);
        let sub = SubTexture::region(3, Vec2::new(0.25, 0.25), Vec2::splat(0.5));
        b.push_quad(Vec2::ZERO, Vec2::splat(10.0), white(), &sub)
            .unwrap();
        let v = b.vertices();
        assert_eq!(v[0].uv, Vec2::new(0.25, 0.25), "top-left of the sub-region");
        assert_eq!(
            v[2].uv,
            Vec2::new(0.75, 0.75),
            "bottom-right of the sub-region"
        );
        assert!(v.iter().all(|q| q.texture_index == 3));
    }

    #[test]
    fn the_texture_index_travels_with_the_quad() {
        let mut b = Batcher::new(4);
        b.push_quad(Vec2::ZERO, Vec2::ONE, white(), &SubTexture::full(7))
            .unwrap();
        assert!(b.vertices().iter().all(|q| q.texture_index == 7));
    }

    #[test]
    fn color_travels_with_every_vertex() {
        let mut b = Batcher::new(4);
        b.push_quad(Vec2::ZERO, Vec2::ONE, [1, 2, 3, 4], &SubTexture::full(0))
            .unwrap();
        assert!(b.vertices().iter().all(|q| q.color == [1, 2, 3, 4]));
    }

    #[test]
    fn many_quads_accumulate() {
        let mut b = Batcher::new(64);
        for i in 0..10 {
            b.push_quad(
                Vec2::new(i as f32, 0.0),
                Vec2::ONE,
                white(),
                &SubTexture::full(0),
            )
            .unwrap();
        }
        assert_eq!(b.len(), 10);
        assert_eq!(b.vertex_count(), 60);
    }

    #[test]
    fn a_run_of_copies_shares_one_quad() {
        let mut b = Batcher::new(128);
        b.push_quads(Vec2::ZERO, Vec2::ONE, white(), &SubTexture::full(0), 100)
            .unwrap();
        assert_eq!(b.len(), 100);
        assert_eq!(b.vertex_count(), 600);
    }

    #[test]
    fn overfilling_is_an_error_not_a_silent_drop() {
        let mut b = Batcher::new(2);
        b.push_quad(Vec2::ZERO, Vec2::ONE, white(), &SubTexture::full(0))
            .unwrap();
        b.push_quad(Vec2::ZERO, Vec2::ONE, white(), &SubTexture::full(0))
            .unwrap();
        let err = b
            .push_quad(Vec2::ZERO, Vec2::ONE, white(), &SubTexture::full(0))
            .unwrap_err();
        assert!(
            matches!(err, RenderError::BatchTooLarge { capacity: 2, .. }),
            "{err:?}"
        );
        assert_eq!(b.len(), 2, "the rejected quad must not be half-added");
    }

    #[test]
    fn a_whole_run_is_rejected_rather_than_partially_added() {
        let mut b = Batcher::new(4);
        b.push_quads(Vec2::ZERO, Vec2::ONE, white(), &SubTexture::full(0), 3)
            .unwrap();
        assert!(
            b.push_quads(Vec2::ZERO, Vec2::ONE, white(), &SubTexture::full(0), 2)
                .is_err()
        );
        assert_eq!(b.len(), 3, "a rejected run must add nothing");
    }

    #[test]
    fn a_rotated_quad_stays_the_same_size() {
        let mut b = Batcher::new(4);
        b.push_rotated(
            Vec2::ZERO,
            Vec2::new(10.0, 4.0),
            std::f32::consts::FRAC_PI_2,
            white(),
            &SubTexture::full(0),
        )
        .unwrap();
        let v = b.vertices();
        // A quarter turn carries the 10-wide edge onto the vertical axis, so
        // the edge to vertex 1 measures 10 and the edge to vertex 5 measures 4.
        let first_edge = (v[1].position - v[0].position).length();
        let second_edge = (v[5].position - v[0].position).length();
        assert!(
            (first_edge - 10.0).abs() < 1e-4,
            "width rotated into vertical: {first_edge}"
        );
        assert!(
            (second_edge - 4.0).abs() < 1e-4,
            "height rotated into horizontal: {second_edge}"
        );
    }

    #[test]
    fn a_zero_rotation_matches_an_axis_aligned_quad() {
        let mut a = Batcher::new(4);
        let mut b = Batcher::new(4);
        a.push_quad(
            Vec2::new(5.0, 5.0),
            Vec2::new(10.0, 10.0),
            white(),
            &SubTexture::full(2),
        )
        .unwrap();
        b.push_rotated(
            Vec2::new(5.0, 5.0),
            Vec2::new(10.0, 10.0),
            0.0,
            white(),
            &SubTexture::full(2),
        )
        .unwrap();
        assert_eq!(a.vertices(), b.vertices());
    }

    #[test]
    fn a_rotated_quad_overflows_like_a_plain_one() {
        let mut b = Batcher::new(1);
        b.push_rotated(Vec2::ZERO, Vec2::ONE, 1.0, white(), &SubTexture::full(0))
            .unwrap();
        assert!(
            b.push_rotated(Vec2::ZERO, Vec2::ONE, 1.0, white(), &SubTexture::full(0))
                .is_err()
        );
    }

    #[test]
    fn clear_empties_but_keeps_the_capacity() {
        let mut b = Batcher::new(32);
        b.push_quads(Vec2::ZERO, Vec2::ONE, white(), &SubTexture::full(0), 10)
            .unwrap();
        b.mark_flushed();
        b.clear();
        assert!(b.is_empty());
        assert_eq!(b.capacity(), 32);
        assert_eq!(b.batches, 0);
    }

    #[test]
    fn byte_length_is_six_per_quad() {
        let mut b = Batcher::new(8);
        b.push_quads(Vec2::ZERO, Vec2::ONE, white(), &SubTexture::full(0), 3)
            .unwrap();
        assert_eq!(b.byte_len(), 18 * std::mem::size_of::<QuadVertex>());
    }

    #[test]
    fn stats_report_the_frame_cost() {
        let mut b = Batcher::new(8);
        b.push_quads(Vec2::ZERO, Vec2::ONE, white(), &SubTexture::full(0), 4)
            .unwrap();
        b.mark_flushed();
        let s = b.stats();
        assert_eq!(s.quads, 4);
        assert_eq!(s.draw_calls, 1);
        assert_eq!(s.vertices, 24);
        assert_eq!(s.culled, 0);
    }

    #[test]
    fn culling_keeps_a_quad_that_overlaps() {
        let visible = Aabb2::new(Vec2::ZERO, Vec2::new(100.0, 100.0));
        assert!(is_quad_visible(
            Aabb2::new(Vec2::new(50.0, 50.0), Vec2::new(60.0, 60.0)),
            visible
        ));
    }

    #[test]
    fn culling_drops_a_quad_entirely_left_of_the_view() {
        let visible = Aabb2::new(Vec2::ZERO, Vec2::new(100.0, 100.0));
        assert!(!is_quad_visible(
            Aabb2::new(Vec2::new(-50.0, 0.0), Vec2::new(-10.0, 10.0)),
            visible
        ));
    }

    #[test]
    fn culling_drops_a_quad_entirely_below_the_view() {
        let visible = Aabb2::new(Vec2::ZERO, Vec2::new(100.0, 100.0));
        assert!(!is_quad_visible(
            Aabb2::new(Vec2::new(0.0, 200.0), Vec2::new(10.0, 210.0)),
            visible
        ));
    }

    #[test]
    fn culling_keeps_a_quad_touching_the_edge() {
        let visible = Aabb2::new(Vec2::ZERO, Vec2::new(100.0, 100.0));
        assert!(is_quad_visible(
            Aabb2::new(Vec2::new(0.0, 0.0), Vec2::splat(1.0)),
            visible
        ));
    }

    #[test]
    fn a_vertex_is_exactly_thirty_two_bytes() {
        // The batcher's stride and the pipeline's vertex layout both depend on
        // this; a drift would read the wrong bytes with no error.
        assert_eq!(std::mem::size_of::<QuadVertex>(), 32);
    }
}
