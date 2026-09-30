//! The viewport: the scene area, and the transform gizmo drawn over it.
//!
//! The gizmo is painted with egui's own shapes rather than through a second
//! renderer, and that is the point. egui already emits a flat list of coloured
//! triangles clipped to a rectangle, which is exactly what the UI pipeline
//! draws, so a gizmo drawn as egui shapes goes out in the same batches as the
//! panels with no extra pipeline, no extra vertex buffer and no draw-call
//! accounting to keep in step. The maths for *where* the handles are and *which
//! one the cursor is on* lives in [`crate::gizmo`]; this module is only the
//! painting and the click handling.

use egui::{Color32, Pos2, Rect, Sense, Stroke, Ui};
use glam::{Mat4, Vec2, Vec3};

use crate::camera::EditorCamera;
use crate::gizmo::{GizmoAxis, GizmoDrag, ProjectedHandle, pick, project_handles};

/// How long a gizmo handle is on screen, in pixels.
///
// A constant screen length rather than a world one, so the gizmo stays
// grabbable at any zoom: a world-fixed handle is a dot up close and a
// screen-filling smear from far away.
pub const HANDLE_PIXELS: f32 = 80.0;

/// The colour each axis is drawn in, matching the convention every 3D editor
/// uses so a handle is recognisable without reading a label.
pub const AXIS_COLORS: [(GizmoAxis, Color32); 3] = [
    (GizmoAxis::X, Color32::from_rgb(230, 78, 78)),
    (GizmoAxis::Y, Color32::from_rgb(120, 210, 90)),
    (GizmoAxis::Z, Color32::from_rgb(80, 140, 235)),
];

/// The colour a hovered or dragged handle takes.
const HIGHLIGHT: Color32 = Color32::from_rgb(255, 220, 90);

/// The colour the gizmo's origin marker is drawn in.
const ORIGIN: Color32 = Color32::from_rgb(220, 220, 220);

/// How wide a handle's line is, in pixels.
const HANDLE_WIDTH: f32 = 4.0;

/// How big the tip dot at the end of a handle is, in pixels.
const TIP_RADIUS: f32 = 5.0;

/// How big the origin dot is, in pixels.
const ORIGIN_RADIUS: f32 = 4.0;

/// What the viewport is showing and what the user is doing to it.
#[derive(Debug, Clone)]
pub struct Viewport {
    /// Whether the gizmo is drawn at all.
    pub show_gizmo: bool,
    /// The world position the gizmo sits at.
    pub gizmo_origin: Vec3,
    /// The rotation the gizmo's axes are drawn with.
    pub gizmo_rotation: Mat4,
    /// The drag in progress, if any.
    pub drag: Option<GizmoDrag>,
    /// The axis the cursor is over, for highlighting.
    pub hovered: GizmoAxis,
    /// The rectangle the viewport occupies, in points.
    pub rect: Rect,
    /// How many handles the last draw produced.
    pub handle_count: usize,
    /// The axis the last click grabbed, for the status line.
    pub last_grabbed: GizmoAxis,
}

impl Default for Viewport {
    fn default() -> Self {
        Self::new()
    }
}

impl Viewport {
    /// A viewport with a gizmo at the origin and no drag.
    pub fn new() -> Viewport {
        Viewport {
            show_gizmo: true,
            gizmo_origin: Vec3::ZERO,
            gizmo_rotation: Mat4::IDENTITY,
            drag: None,
            hovered: GizmoAxis::None,
            rect: Rect::ZERO,
            handle_count: 0,
            last_grabbed: GizmoAxis::None,
        }
    }

    /// Place the gizmo on an entity's transform.
    pub fn follow(&mut self, position: Vec3, rotation: Mat4) {
        self.gizmo_origin = position;
        self.gizmo_rotation = rotation;
    }

    /// The handles as they appear on screen, or none when the gizmo is behind
    /// the camera.
    pub fn handles(&self, camera: &EditorCamera, size: Vec2) -> Vec<ProjectedHandle> {
        if !self.show_gizmo {
            return Vec::new();
        }
        project_handles(self.gizmo_origin, camera, size, HANDLE_PIXELS)
    }

    /// The axis a click at `cursor` would grab, given the current handles.
    pub fn axis_at(&self, camera: &EditorCamera, size: Vec2, cursor: Pos2) -> GizmoAxis {
        pick(&self.handles(camera, size), Vec2::new(cursor.x, cursor.y))
    }

    /// Handle a click: start a drag on whatever is under the cursor.
    ///
    /// Returns the axis grabbed, or `None` when the click missed every handle —
    /// which is not an error but the common case, since most clicks in a
    /// viewport are not on a gizmo.
    pub fn begin_drag(&mut self, camera: &EditorCamera, size: Vec2, cursor: Pos2) -> GizmoAxis {
        let axis = self.axis_at(camera, size, cursor);
        if axis == GizmoAxis::None {
            return GizmoAxis::None;
        }
        self.drag = GizmoDrag::begin(axis, self.gizmo_origin, self.gizmo_origin);
        self.last_grabbed = axis;
        axis
    }

    /// Move a drag to a cursor, returning the distance travelled along the axis.
    pub fn update_drag(&mut self, cursor: Pos2) -> f32 {
        let Some(drag) = self.drag.as_mut() else {
            return 0.0;
        };
        // A screen-space cursor has no world position without a ray, so the
        // drag is driven by the axis-projected component of the cursor's offset
        // from the handle's origin. That is enough to move an entity along one
        // axis, which is all a single-axis gizmo offers.
        let offset = Vec3::new(cursor.x as f32, cursor.y as f32, 0.0);
        let Some(dir) = drag.axis.direction() else {
            return 0.0;
        };
        drag.update(self.gizmo_origin + dir * offset.dot(dir))
    }

    /// End a drag, returning the distance travelled so the caller can apply it.
    pub fn end_drag(&mut self) -> Option<(GizmoAxis, f32)> {
        let drag = self.drag.take()?;
        Some((drag.axis, drag.delta))
    }

    /// True while a drag is in progress.
    pub fn is_dragging(&self) -> bool {
        self.drag.is_some()
    }

    /// The colour a handle is drawn in, highlighted when hovered or dragged.
    pub fn color_for(&self, axis: GizmoAxis) -> Color32 {
        let active = axis != GizmoAxis::None
            && (self.hovered == axis || self.drag.map(|d| d.axis) == Some(axis));
        if active {
            return HIGHLIGHT;
        }
        AXIS_COLORS
            .iter()
            .find(|(a, _)| *a == axis)
            .map(|(_, c)| *c)
            .unwrap_or(ORIGIN)
    }

    /// Draw the viewport's gizmo and handle the pointer.
    ///
    /// The gizmo is painted rather than laid out, so it costs no space and does
    /// not push the panels around; the whole viewport is one sense-checking
    /// rect that takes clicks.
    pub fn show(&mut self, ui: &mut Ui, camera: &EditorCamera) {
        let (response, painter) =
            ui.allocate_painter(ui.available_size_before_wrap(), Sense::click_and_drag());
        self.rect = response.rect;
        let size = Vec2::new(response.rect.width(), response.rect.height());

        let pointer = response.hover_pos().unwrap_or(Pos2::ZERO);
        self.hovered = if response.hovered() && !self.is_dragging() {
            self.axis_at(camera, size, pointer)
        } else {
            GizmoAxis::None
        };

        if response.drag_started() {
            self.begin_drag(camera, size, pointer);
        }
        if response.dragged() {
            self.update_drag(pointer);
        }
        if response.drag_stopped() {
            self.end_drag();
        }

        let handles = self.handles(camera, size);
        self.handle_count = handles.len();
        if handles.is_empty() {
            return;
        }

        // A dark outline under each handle first, so a handle stays visible
        // against both a light scene and a dark one.
        for handle in &handles {
            painter.add(egui::Shape::line_segment(
                [
                    Pos2::new(handle.origin.x, handle.origin.y),
                    Pos2::new(handle.tip.x, handle.tip.y),
                ],
                Stroke::new(HANDLE_WIDTH + 2.0, Color32::from_black_alpha(140)),
            ));
        }
        for handle in &handles {
            let color = self.color_for(handle.axis);
            painter.add(egui::Shape::line_segment(
                [
                    Pos2::new(handle.origin.x, handle.origin.y),
                    Pos2::new(handle.tip.x, handle.tip.y),
                ],
                Stroke::new(HANDLE_WIDTH, color),
            ));
            painter.add(egui::Shape::circle_filled(
                Pos2::new(handle.tip.x, handle.tip.y),
                TIP_RADIUS,
                color,
            ));
        }
        if let Some(first) = handles.first() {
            painter.add(egui::Shape::circle_filled(
                Pos2::new(first.origin.x, first.origin.y),
                ORIGIN_RADIUS,
                ORIGIN,
            ));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn camera() -> EditorCamera {
        let mut c = EditorCamera::perspective();
        c.position = Vec3::new(6.0, 5.0, 6.0);
        c
    }

    fn size() -> Vec2 {
        Vec2::new(800.0, 600.0)
    }

    #[test]
    fn a_new_viewport_has_a_gizmo_and_no_drag() {
        let v = Viewport::new();
        assert!(v.show_gizmo);
        assert!(!v.is_dragging());
        assert_eq!(v.hovered, GizmoAxis::None);
    }

    #[test]
    fn the_gizmo_produces_three_handles() {
        let v = Viewport::new();
        assert_eq!(v.handles(&camera(), size()).len(), 3);
    }

    #[test]
    fn hiding_the_gizmo_produces_no_handles() {
        let mut v = Viewport::new();
        v.show_gizmo = false;
        assert!(v.handles(&camera(), size()).is_empty());
    }

    #[test]
    fn following_moves_the_gizmo() {
        let mut v = Viewport::new();
        v.follow(Vec3::new(1.0, 2.0, 3.0), Mat4::IDENTITY);
        assert_eq!(v.gizmo_origin, Vec3::new(1.0, 2.0, 3.0));
    }

    /// A camera on the diagonal looking at the origin, so a gizmo there lands
    /// in the middle of the viewport.
    fn looking_at_origin() -> EditorCamera {
        let mut c = camera();
        c.yaw = std::f32::consts::FRAC_PI_4;
        c.pitch = -0.6;
        c
    }

    #[test]
    fn the_origin_projects_near_the_viewport_centre() {
        // A camera at (6,5,6) with no yaw does not look at the origin, so the
        // gizmo lands off to one side; a camera aimed at it puts the gizmo in the
        // middle, which is the case worth pinning.
        let v = Viewport::new();
        let handles = v.handles(&looking_at_origin(), size());
        assert_eq!(
            handles.len(),
            3,
            "the gizmo must be on screen to be centred"
        );
        let mid = Vec2::new(size().x * 0.5, size().y * 0.5);
        assert!(
            handles[0].origin.distance(mid) < size().x * 0.25,
            "{:?} should be near {mid:?}",
            handles[0].origin
        );
    }

    #[test]
    fn a_cursor_on_a_handle_picks_that_axis() {
        let v = Viewport::new();
        let handles = v.handles(&camera(), size());
        let h = handles[0];
        let mid = (h.origin + h.tip) * 0.5;
        let axis = v.axis_at(&camera(), size(), Pos2::new(mid.x, mid.y));
        assert_eq!(axis, h.axis);
    }

    #[test]
    fn a_cursor_far_from_the_gizmo_picks_nothing() {
        let v = Viewport::new();
        assert_eq!(
            v.axis_at(&camera(), size(), Pos2::new(5.0, 5.0)),
            GizmoAxis::None
        );
    }

    #[test]
    fn a_click_on_a_handle_starts_a_drag() {
        let mut v = Viewport::new();
        let handles = v.handles(&camera(), size());
        let h = handles[0];
        let mid = (h.origin + h.tip) * 0.5;
        let axis = v.begin_drag(&camera(), size(), Pos2::new(mid.x, mid.y));
        assert_eq!(axis, h.axis);
        assert!(v.is_dragging());
        assert_eq!(v.last_grabbed, h.axis);
    }

    #[test]
    fn a_click_off_the_gizmo_starts_nothing() {
        let mut v = Viewport::new();
        let axis = v.begin_drag(&camera(), size(), Pos2::new(3.0, 3.0));
        assert_eq!(axis, GizmoAxis::None);
        assert!(!v.is_dragging());
    }

    #[test]
    fn ending_a_drag_reports_the_axis_and_distance() {
        let mut v = Viewport::new();
        let handles = v.handles(&camera(), size());
        let h = handles[0];
        let mid = (h.origin + h.tip) * 0.5;
        v.begin_drag(&camera(), size(), Pos2::new(mid.x, mid.y));
        v.update_drag(Pos2::new(mid.x + 30.0, mid.y));
        let ended = v.end_drag();
        assert!(ended.is_some());
        assert!(!v.is_dragging(), "the drag is over");
    }

    #[test]
    fn ending_a_drag_that_never_began_is_none() {
        let mut v = Viewport::new();
        assert!(v.end_drag().is_none());
    }

    #[test]
    fn a_drag_on_a_hidden_gizmo_cannot_begin() {
        let mut v = Viewport::new();
        v.show_gizmo = false;
        assert_eq!(
            v.begin_drag(&camera(), size(), Pos2::new(400.0, 300.0)),
            GizmoAxis::None
        );
    }

    #[test]
    fn every_axis_has_its_own_colour() {
        let v = Viewport::new();
        let colors: Vec<Color32> = GizmoAxis::singles()
            .iter()
            .map(|a| v.color_for(*a))
            .collect();
        for (i, a) in colors.iter().enumerate() {
            for b in colors.iter().skip(i + 1) {
                assert_ne!(a, b, "two axes share a colour");
            }
        }
    }

    #[test]
    fn a_hovered_axis_is_highlighted() {
        let mut v = Viewport::new();
        let resting = v.color_for(GizmoAxis::X);
        v.hovered = GizmoAxis::X;
        assert_ne!(v.color_for(GizmoAxis::X), resting);
    }

    #[test]
    fn a_dragged_axis_is_highlighted() {
        let mut v = Viewport::new();
        // The resting colour has to be read before the drag exists, or the
        // comparison is a colour against itself.
        let resting = v.color_for(GizmoAxis::Y);
        v.drag = GizmoDrag::begin(GizmoAxis::Y, Vec3::ZERO, Vec3::ZERO);
        assert_ne!(v.color_for(GizmoAxis::Y), resting);
    }

    #[test]
    fn an_unhovered_axis_keeps_its_own_colour() {
        let v = Viewport::new();
        assert_eq!(v.color_for(GizmoAxis::Z), AXIS_COLORS[2].1);
    }

    #[test]
    fn the_handle_length_is_a_constant_in_pixels() {
        let v = Viewport::new();
        for h in v.handles(&camera(), size()) {
            assert!(
                (h.length() - HANDLE_PIXELS).abs() < 1.0,
                "{:?} is {} px",
                h.axis,
                h.length()
            );
        }
    }

    #[test]
    fn a_gizmo_behind_the_camera_has_no_handles() {
        let mut c = camera();
        // Put the camera where the gizmo is, so it is at the near plane.
        c.position = Vec3::ZERO;
        let v = Viewport::new();
        assert!(v.handles(&c, size()).is_empty());
    }
}
