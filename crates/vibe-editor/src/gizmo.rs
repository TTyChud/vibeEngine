//! Transform gizmos: picking an axis handle, and dragging an entity along it.
//!
//! The maths here is the gizmo. Drawing the handles is a loop over three axes
//! and a painter call, and a mistake in it is visible immediately. A mistake in
//! the picking is not: a ray that is a few pixels off selects the wrong axis,
//! and dragging then moves the entity in a direction the user did not choose.
//! So the projection and the closest-point-on-a-line are the parts worth
//! testing, and both are pure functions of a camera and a cursor.

use glam::{Mat3, Mat4, Vec2, Vec3, Vec4};

use crate::camera::EditorCamera;

/// Which handle the user grabbed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum GizmoAxis {
    /// The X axis.
    X,
    /// The Y axis.
    Y,
    /// The Z axis.
    Z,
    /// The screen-plane handle, which moves in two axes at once.
    Screen,
    /// No handle: the click missed.
    #[default]
    None,
}

impl GizmoAxis {
    /// The axis's index, for matching a drag delta to a transform field.
    pub fn index(self) -> Option<usize> {
        match self {
            GizmoAxis::X => Some(0),
            GizmoAxis::Y => Some(1),
            GizmoAxis::Z => Some(2),
            GizmoAxis::Screen | GizmoAxis::None => None,
        }
    }

    /// The unit vector this axis points along, in the object's space.
    pub fn direction(self) -> Option<Vec3> {
        self.index().map(|i| {
            let mut d = Vec3::ZERO;
            d[i] = 1.0;
            d
        })
    }

    /// The axis's name, for the status bar.
    pub fn label(self) -> &'static str {
        match self {
            GizmoAxis::X => "X",
            GizmoAxis::Y => "Y",
            GizmoAxis::Z => "Z",
            GizmoAxis::Screen => "Screen",
            GizmoAxis::None => "",
        }
    }

    /// The three single-axis handles, in drawing order.
    pub fn singles() -> [GizmoAxis; 3] {
        [GizmoAxis::X, GizmoAxis::Y, GizmoAxis::Z]
    }
}

/// A handle as it appears on screen: where it starts, where it ends, and how
/// thick it is.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ProjectedHandle {
    /// Which axis this is.
    pub axis: GizmoAxis,
    /// The handle's origin, in pixels.
    pub origin: Vec2,
    /// The handle's tip, in pixels.
    pub tip: Vec2,
    /// How many pixels from the line counts as a grab.
    pub pick_radius: f32,
}

impl ProjectedHandle {
    /// The handle's length in pixels.
    pub fn length(&self) -> f32 {
        self.tip.distance(self.origin)
    }

    /// True when a cursor within `radius` of the line grabs this handle.
    pub fn contains(&self, cursor: Vec2, radius: f32) -> bool {
        distance_to_segment(cursor, self.origin, self.tip) <= radius.max(self.pick_radius)
    }
}

/// Project a world-space point into screen pixels.
///
/// A point behind the camera has no screen position; it is reported as `None`
/// rather than projected with a negative w, which would mirror the point to the
/// opposite side of the screen and let the user grab a handle that is behind
/// them.
pub fn project(point: Vec3, view_projection: Mat4, viewport: Vec2) -> Option<Vec2> {
    let clip = view_projection * Vec4::new(point.x, point.y, point.z, 1.0);
    if clip.w <= f32::EPSILON {
        return None;
    }
    let ndc = clip.truncate() / clip.w;
    Some(Vec2::new(
        (ndc.x * 0.5 + 0.5) * viewport.x,
        // NDC's +Y is up and the screen's is down, so the sign flips here.
        (1.0 - (ndc.y * 0.5 + 0.5)) * viewport.y,
    ))
}

/// Project a world point with a camera and a viewport.
pub fn project_with_camera(point: Vec3, camera: &EditorCamera, viewport: Vec2) -> Option<Vec2> {
    let vp = camera.view_projection(viewport.x / viewport.y.max(1.0));
    project(point, vp, viewport)
}

/// Project the three axis handles of a gizmo at a world position.
///
/// The axes are scaled to a constant screen length rather than a constant world
/// length, so a gizmo stays grabbable whether the camera is close or far. That
/// needs the camera's distance to the gizmo, which is why this takes a camera
/// and not just a matrix.
pub fn project_handles(
    origin: Vec3,
    camera: &EditorCamera,
    viewport: Vec2,
    pixel_length: f32,
) -> Vec<ProjectedHandle> {
    let aspect = viewport.x / viewport.y.max(1.0);
    let vp = camera.view_projection(aspect);
    let Some(origin_px) = project(origin, vp, viewport) else {
        return Vec::new();
    };
    // A handle is a fixed number of *pixels* long, so its world length is not
    // simply proportional to the distance: the projection already shrinks with
    // distance, so the world length has to be the pixel length times the
    // distance and then divided by the projection's own scale — which for a
    // perspective matrix is the distance itself. The two cancel, leaving a
    // world length that is independent of distance, and multiplying by the
    // distance as well is what makes a far handle thousands of pixels long.
    //
    // What is actually needed is the world length whose *projected* length is
    // pixel_length. Measuring that directly, rather than deriving it, keeps the
    // constant honest if the camera's fov or aspect ever changes.
    //
    // Measured per axis, and that is the part that is easy to get wrong: a
    // handle pointing at the camera is foreshortened by perspective, so the
    // world length that projects to 80 pixels along X is a different one along
    // Z. Reusing X's length for all three draws the near-axis handle long and
    // the far-axis one short, which looks like the gizmo is sheared.
    GizmoAxis::singles()
        .into_iter()
        .filter_map(|axis| {
            let dir = axis.direction()?;
            let world_length = world_length_for_pixels(&vp, origin, viewport, pixel_length, dir);
            let tip_world = origin + dir * world_length;
            let tip = project(tip_world, vp, viewport)?;
            Some(ProjectedHandle {
                axis,
                origin: origin_px,
                tip,
                // Generous relative to the handle: a 3-pixel line is hard to hit
                // with a mouse, and a gizmo that cannot be grabbed is worse
                // than one that is slightly sticky.
                pick_radius: 6.0,
            })
        })
        .collect()
}

/// The world-space length whose projected length is about `pixels`.
///
/// Found by measuring rather than derived. The projection's scale depends on
/// the field of view, the aspect, the distance and the direction, and a formula
/// that gets one of them wrong produces a handle that is the right length in
/// one direction and wildly wrong in another — which is exactly the bug this
/// replaces. A bisection on the measured length cannot be wrong that way.
///
/// `direction` must be a unit vector: the handle is measured along it, so a
/// scaled direction would return a length scaled with it.
pub fn world_length_for_pixels(
    view_projection: &Mat4,
    origin: Vec3,
    viewport: Vec2,
    pixels: f32,
    direction: Vec3,
) -> f32 {
    let measure = |len: f32| -> Option<f32> {
        let o = project(origin, *view_projection, viewport)?;
        let tip = project(origin + direction * len, *view_projection, viewport)?;
        Some(tip.distance(o))
    };
    if pixels <= 0.0 || measure(0.0).is_none() {
        return 0.0;
    }
    // Grow the upper bound until the projected length passes the target, then
    // bisect. A handle is at most a few hundred world units, so this converges
    // in well under thirty steps and the guard stops a pathological camera from
    // spinning forever.
    let mut lo = 1e-4f32;
    let mut hi = 1.0f32;
    for _ in 0..64 {
        match measure(hi) {
            Some(l) if l < pixels => hi *= 2.0,
            _ => break,
        }
    }
    for _ in 0..40 {
        let mid = (lo + hi) * 0.5;
        match measure(mid) {
            Some(l) if l < pixels => lo = mid,
            Some(_) => hi = mid,
            None => return hi,
        }
    }
    (lo + hi) * 0.5
}

/// Pick the handle nearest a cursor.
///
/// Ties go to the earlier axis, so the choice is stable between frames: a
/// cursor sitting exactly between two handles must not alternate between them
/// as the mouse jitters, which would make a drag stutter between directions.
pub fn pick(handles: &[ProjectedHandle], cursor: Vec2) -> GizmoAxis {
    let mut best = GizmoAxis::None;
    let mut best_distance = f32::INFINITY;
    for handle in handles {
        let d = distance_to_segment(cursor, handle.origin, handle.tip);
        if d <= handle.pick_radius && d < best_distance {
            best_distance = d;
            best = handle.axis;
        }
    }
    best
}

/// The distance from a point to a line segment.
pub fn distance_to_segment(p: Vec2, a: Vec2, b: Vec2) -> f32 {
    let ab = b - a;
    let len_sq = ab.length_squared();
    let t = if len_sq <= f32::EPSILON {
        0.0
    } else {
        ((p - a).dot(ab) / len_sq).clamp(0.0, 1.0)
    };
    p.distance(a + ab * t)
}

/// The closest point on a line through `origin` along `direction` to a world
/// point.
pub fn closest_point_on_line(origin: Vec3, direction: Vec3, point: Vec3) -> Vec3 {
    let dir = direction.normalize_or_zero();
    if dir == Vec3::ZERO {
        return origin;
    }
    origin + dir * (point - origin).dot(dir)
}

/// How far a world point has moved along an axis since a drag began.
///
/// This is the whole of a gizmo drag: the delta is measured against the
/// *original* position rather than accumulated per frame, so a dropped frame or
/// a reordered input cannot make the entity drift away from the cursor.
pub fn axis_delta(
    drag_start_point: Vec3,
    drag_start_origin: Vec3,
    direction: Vec3,
    current_point: Vec3,
) -> f32 {
    let dir = direction.normalize_or_zero();
    if dir == Vec3::ZERO {
        return 0.0;
    }
    (closest_point_on_line(drag_start_origin, dir, current_point)
        - closest_point_on_line(drag_start_origin, dir, drag_start_point))
    .dot(dir)
}

/// A drag in progress.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GizmoDrag {
    /// Which handle is being dragged.
    pub axis: GizmoAxis,
    /// Where the drag began, in world space.
    pub start_origin: Vec3,
    /// Where the cursor was when the drag began, projected onto the axis.
    pub start_point: Vec3,
    /// The last value the drag produced.
    pub delta: f32,
}

impl GizmoDrag {
    /// Begin a drag on a handle.
    ///
    /// Returns `None` for a handle that names no axis, because there is no
    /// direction to constrain a drag to.
    pub fn begin(axis: GizmoAxis, origin: Vec3, point: Vec3) -> Option<GizmoDrag> {
        axis.direction()?;
        Some(GizmoDrag {
            axis,
            start_origin: origin,
            start_point: point,
            delta: 0.0,
        })
    }

    /// The distance travelled so far along the axis.
    pub fn update(&mut self, point: Vec3) -> f32 {
        let Some(dir) = self.axis.direction() else {
            return self.delta;
        };
        self.delta = axis_delta(self.start_point, self.start_origin, dir, point);
        self.delta
    }

    /// True while the drag is in progress.
    pub fn is_active(&self) -> bool {
        self.axis != GizmoAxis::None
    }
}

/// A ray in world space: where it starts and the direction it travels.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Ray {
    /// A point on the ray, normally the camera's position.
    pub origin: Vec3,
    /// The unit direction the ray travels in.
    pub direction: Vec3,
}

impl Ray {
    /// The point at `distance` along the ray.
    pub fn at(&self, distance: f32) -> Vec3 {
        self.origin + self.direction * distance
    }

    /// Where this ray meets a plane through `point` with the given normal.
    ///
    /// `None` when the ray is parallel to the plane, or points away from it. Both
    /// are ordinary situations rather than errors: a resize drag whose ray never
    /// crosses the plane the handle lives in has no answer, and returning `None`
    /// lets the caller leave the box where it is instead of snapping it to
    /// infinity.
    pub fn intersect_plane(&self, point: Vec3, normal: Vec3) -> Option<Vec3> {
        let denom = self.direction.dot(normal);
        // A ray almost parallel to the plane gives a wildly distant
        // intersection, so a small denominator is treated as parallel rather than
        // divided by. The threshold is on the cosine, so it is a real angle.
        if denom.abs() < 1.0e-4 {
            return None;
        }
        let t = (point - self.origin).dot(normal) / denom;
        if t < 0.0 {
            return None;
        }
        Some(self.at(t))
    }
}

/// The ray through a screen position, from a camera and a viewport.
///
/// This is the inverse of [`project`], and it is what turns a mouse position into
/// a place in the scene. Resizing a box in three dimensions cannot be driven from
/// screen pixels alone: the cursor has to be turned into a world position, and
/// the only correct way to do that is to undo the same matrices the projection
/// applied.
///
/// The ray is built from two unprojected points rather than from a single
/// unprojected one, because a point unprojected at NDC depth 0 lands on the near
/// plane and says nothing about direction. Two depths give a line, and the
/// direction is the difference between them normalised.
///
/// The Y flip is undone here explicitly rather than being left to the matrix
/// inverse, so that this and [`project`] are visibly a pair: a ray that does not
/// come back out under the pointer is the signature of the two disagreeing about
/// which way is up.
pub fn screen_ray(cursor: Vec2, view_projection: Mat4, viewport: Vec2) -> Option<Ray> {
    if viewport.x <= 0.0 || viewport.y <= 0.0 {
        return None;
    }
    // Screen pixels to NDC, undoing the Y flip `project` applies.
    let ndc_x = (cursor.x / viewport.x) * 2.0 - 1.0;
    let ndc_y = 1.0 - (cursor.y / viewport.y) * 2.0;
    let unproject = |ndc: Vec3| {
        let clip = view_projection.inverse() * ndc.extend(1.0);
        // A point at infinity comes back with a w near zero, which is not a
        // position and would make the direction meaningless.
        if !clip.w.is_finite() || clip.w.abs() < 1.0e-6 {
            return None;
        }
        Some(clip.truncate() / clip.w)
    };
    // Two depths, so the ray is a direction and not a single point. The near
    // plane is negative because the view space is right-handed with what is in
    // front at -Z, which is the same convention the projection is built for.
    let near = unproject(Vec3::new(ndc_x, ndc_y, 0.0))?;
    let far = unproject(Vec3::new(ndc_x, ndc_y, 1.0))?;
    let direction = (far - near).normalize_or_zero();
    if direction == Vec3::ZERO {
        return None;
    }
    Some(Ray {
        origin: near,
        direction,
    })
}

/// The ray through a screen position, from a camera and a viewport.
pub fn screen_ray_with_camera(cursor: Vec2, camera: &EditorCamera, viewport: Vec2) -> Option<Ray> {
    let aspect = viewport.x / viewport.y.max(1.0);
    screen_ray(cursor, camera.view_projection(aspect), viewport)
}

/// Where a ray meets an axis-aligned box, or `None` if it misses.
///
/// The slab method: the ray enters the box where it crosses the near face of
/// each axis and leaves at the far face, and it hits only if every axis is
/// crossed within the same span. A ray that misses on any one axis misses the
/// box, which is what makes this exact rather than a per-face test that has to
/// be right about all six.
///
/// Returns the *near* intersection, so a click selects the face the user can
/// actually see rather than the far side of the box.
pub fn ray_box_intersection(
    origin: Vec3,
    direction: Vec3,
    centre: Vec3,
    size: Vec3,
) -> Option<f32> {
    let half = size.abs() * 0.5;
    let mut t_enter = f32::NEG_INFINITY;
    let mut t_exit = f32::INFINITY;
    for axis in 0..3 {
        // The slab for this axis, as the two planes bounding it.
        let (o, d) = (origin[axis], direction[axis]);
        let (lo, hi) = (centre[axis] - half[axis], centre[axis] + half[axis]);
        if d.abs() < 1.0e-8 {
            // Parallel to this pair of planes: either always inside them, which
            // contributes no constraint, or never, which misses outright.
            if o < lo || o > hi {
                return None;
            }
            continue;
        }
        let t0 = (lo - o) / d;
        let t1 = (hi - o) / d;
        // The two crossings are in whatever order the direction implies, so they
        // are sorted rather than assumed.
        let (near, far) = if t0 < t1 { (t0, t1) } else { (t1, t0) };
        t_enter = t_enter.max(near);
        t_exit = t_exit.min(far);
        // The spans no longer overlap, so the ray has left the box before it
        // finished entering it.
        if t_enter > t_exit {
            return None;
        }
    }
    // The box was crossed on every axis, and the span is consistent. The one
    // thing left is whether that span is ahead of the ray at all: a box entirely
    // behind the origin still produces a consistent span, just a wholly negative
    // one, and crossing its planes is not a hit if the crossing was behind us.
    if t_exit < 0.0 {
        return None;
    }
    // The near crossing, so a click selects the face the user can see. A ray
    // starting inside the box has a negative entry, and the exit is the only
    // surface there is to see from in there.
    Some(if t_enter >= 0.0 { t_enter } else { t_exit })
}

/// The point a ray meets an axis-aligned box, or `None` if it misses.
pub fn ray_hits_box(origin: Vec3, direction: Vec3, centre: Vec3, size: Vec3) -> Option<Vec3> {
    let t = ray_box_intersection(origin, direction, centre, size)?;
    Some(origin + direction * t)
}

/// The rotation of a gizmo, from its object's world transform.
///
/// A gizmo has to rotate with the object it edits, or dragging its local Y on a
/// rotated object moves along the object's Y rather than the world's, which is
/// the whole difference between a local and a world-space gizmo.
pub fn gizmo_rotation(world: Mat4, local_space: bool) -> Mat3 {
    if !local_space {
        return Mat3::IDENTITY;
    }
    // The columns are normalised individually rather than the matrix as a
    // whole: a uniform scale is one multiply away from being removed either
    // way, but a non-uniform one only comes out right per column, and leaving
    // the skew in would make the three handles non-perpendicular.
    let m = Mat3::from_mat4(world);
    Mat3::from_cols(
        m.x_axis.normalize_or_zero(),
        m.y_axis.normalize_or_zero(),
        m.z_axis.normalize_or_zero(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn identity_vp() -> Mat4 {
        Mat4::IDENTITY
    }

    fn axes_have_indices() {
        assert_eq!(GizmoAxis::X.index(), Some(0));
        assert_eq!(GizmoAxis::Y.index(), Some(1));
        assert_eq!(GizmoAxis::Z.index(), Some(2));
        assert_eq!(GizmoAxis::Screen.index(), None);
        assert_eq!(GizmoAxis::None.index(), None);
    }

    #[test]
    fn each_axis_has_a_unit_direction() {
        for axis in GizmoAxis::singles() {
            let d = axis.direction().unwrap();
            assert!((d.length() - 1.0).abs() < 1e-6, "{axis:?} {d:?}");
        }
    }

    #[test]
    fn the_axis_directions_are_the_basis_vectors() {
        assert_eq!(GizmoAxis::X.direction().unwrap(), Vec3::X);
        assert_eq!(GizmoAxis::Y.direction().unwrap(), Vec3::Y);
        assert_eq!(GizmoAxis::Z.direction().unwrap(), Vec3::Z);
    }

    #[test]
    fn the_three_axes_are_perpendicular() {
        let (x, y, z) = (
            GizmoAxis::X.direction().unwrap(),
            GizmoAxis::Y.direction().unwrap(),
            GizmoAxis::Z.direction().unwrap(),
        );
        assert!(x.dot(y).abs() < 1e-6);
        assert!(x.dot(z).abs() < 1e-6);
        assert!(y.dot(z).abs() < 1e-6);
    }

    #[test]
    fn a_screen_handle_has_no_direction() {
        assert!(GizmoAxis::Screen.direction().is_none());
    }

    #[test]
    fn axis_labels_are_distinct() {
        let labels: std::collections::HashSet<_> =
            [GizmoAxis::X, GizmoAxis::Y, GizmoAxis::Z, GizmoAxis::Screen]
                .iter()
                .map(|a| a.label())
                .collect();
        assert_eq!(labels.len(), 4);
    }

    #[test]
    fn projecting_the_origin_lands_in_the_middle() {
        let p = project(Vec3::ZERO, identity_vp(), Vec2::new(800.0, 600.0)).unwrap();
        assert!((p - Vec2::new(400.0, 300.0)).length() < 1e-3, "{p:?}");
    }

    #[test]
    fn projecting_a_point_up_moves_it_up_on_screen() {
        // NDC +Y is up, the screen's is down, so a point above the centre
        // projects to a smaller y. Getting this backwards flips the gizmo's Y
        // handle to point at the floor.
        let p = project(
            Vec3::new(0.0, 0.5, 0.0),
            identity_vp(),
            Vec2::new(800.0, 600.0),
        )
        .unwrap();
        assert!(p.y < 300.0, "up on screen means a smaller y: {p:?}");
    }

    #[test]
    fn projecting_maps_the_corners_to_the_corners() {
        let v = Vec2::new(800.0, 600.0);
        let tl = project(Vec3::new(-1.0, 1.0, 0.0), identity_vp(), v).unwrap();
        assert!((tl - Vec2::new(0.0, 0.0)).length() < 1e-3, "{tl:?}");
        let br = project(Vec3::new(1.0, -1.0, 0.0), identity_vp(), v).unwrap();
        assert!((br - v).length() < 1e-3, "{br:?}");
    }

    /// A camera at the origin looking down -Z, so +Z is behind it.
    fn behind_camera() -> (EditorCamera, Vec2) {
        let mut camera = EditorCamera::perspective();
        camera.position = Vec3::ZERO;
        camera.yaw = 0.0;
        camera.pitch = 0.0;
        (camera, Vec2::new(800.0, 600.0))
    }

    #[test]
    fn a_point_behind_the_camera_is_not_projected() {
        let (camera, viewport) = behind_camera();
        let behind = Vec3::new(0.0, 0.0, 5.0);
        assert!(
            project_with_camera(behind, &camera, viewport).is_none(),
            "a handle behind the camera has no screen position"
        );
    }

    #[test]
    fn a_point_behind_the_camera_is_not_mirrored() {
        // A negative w would flip the point to the opposite side of the screen
        // rather than rejecting it, and the user would grab a handle behind them.
        let (camera, viewport) = behind_camera();
        let behind = Vec3::new(0.0, 0.0, 5.0);
        let p = project_with_camera(behind, &camera, viewport);
        assert!(p.is_none(), "{p:?}");
    }

    #[test]
    fn a_point_in_front_of_the_camera_is_projected() {
        let (camera, viewport) = behind_camera();
        let front = Vec3::new(0.0, 0.0, -5.0);
        assert!(project_with_camera(front, &camera, viewport).is_some());
    }

    /// A camera looking at a gizmo from an angle where all three handles are
    /// visible.
    ///
    /// Two things force the angle. The gizmo cannot be at the origin, because
    /// the default camera sits there and the handles would be at the near plane.
    /// And the camera cannot look straight down -Z, because the Z handle then
    /// points directly away and projects to a point — correct behaviour, but not
    /// what a length assertion wants to measure.
    fn scene() -> (EditorCamera, Vec3, Vec2) {
        let mut camera = EditorCamera::perspective();
        camera.position = Vec3::new(8.0, 6.0, 8.0);
        camera.yaw = 0.0;
        camera.pitch = 0.0;
        (camera, Vec3::ZERO, Vec2::new(800.0, 600.0))
    }

    #[test]
    fn a_handle_has_a_positive_length() {
        let (camera, origin, viewport) = scene();
        let handles = project_handles(origin, &camera, viewport, 80.0);
        assert_eq!(handles.len(), 3, "{handles:?}");
        for h in &handles {
            assert!(h.length() > 1.0, "{:?} is too short", h.axis);
        }
    }

    #[test]
    fn handles_share_an_origin() {
        let (camera, origin, viewport) = scene();
        let handles = project_handles(origin, &camera, viewport, 80.0);
        let first = handles[0].origin;
        for h in &handles[1..] {
            assert!((h.origin - first).length() < 1e-3, "{:?}", h.axis);
        }
    }

    #[test]
    fn a_gizmo_at_zero_length_projects_no_handles() {
        let mut camera = EditorCamera::perspective();
        camera.position = Vec3::ZERO;
        // The camera is inside the gizmo, so the origin is behind it and there
        // is nothing sensible to draw.
        let handles = project_handles(Vec3::ZERO, &camera, Vec2::new(800.0, 600.0), 80.0);
        assert!(handles.is_empty());
    }

    #[test]
    fn a_gizmo_keeps_its_pixel_length_at_any_distance() {
        let viewport = Vec2::new(800.0, 600.0);
        // Two cameras the same distance from the gizmo's *own* plane, one near
        // and one far, both looking down -Z. What is being checked is that the
        // handle's pixel length does not depend on the distance, so the gizmo
        // stays grabbable at any zoom.
        // Both cameras look at the origin from a diagonal, one near and one
        // far. The far one has to extend its far plane or the gizmo is clipped.
        let mut near = EditorCamera::perspective();
        near.position = Vec3::new(4.0, 3.0, 4.0);
        near.far = 1000.0;
        let mut far = EditorCamera::perspective();
        far.position = Vec3::new(40.0, 30.0, 40.0);
        far.far = 4000.0;
        let a = project_handles(Vec3::ZERO, &near, viewport, 80.0);
        let b = project_handles(Vec3::ZERO, &far, viewport, 80.0);
        assert_eq!(a.len(), 3, "{a:?}");
        assert_eq!(b.len(), 3, "{b:?}");
        for (ha, hb) in a.iter().zip(b.iter()) {
            let la = ha.length();
            let lb = hb.length();
            assert!(
                (la - lb).abs() < 1.0,
                "{:?}: {la} near vs {lb} far",
                ha.axis
            );
        }
    }

    #[test]
    fn a_cursor_on_a_handle_picks_it() {
        let h = ProjectedHandle {
            axis: GizmoAxis::X,
            origin: Vec2::new(100.0, 100.0),
            tip: Vec2::new(200.0, 100.0),
            pick_radius: 6.0,
        };
        assert_eq!(pick(&[h], Vec2::new(150.0, 102.0)), GizmoAxis::X);
    }

    #[test]
    fn a_cursor_far_from_every_handle_picks_nothing() {
        let h = ProjectedHandle {
            axis: GizmoAxis::X,
            origin: Vec2::new(100.0, 100.0),
            tip: Vec2::new(200.0, 100.0),
            pick_radius: 6.0,
        };
        assert_eq!(pick(&[h], Vec2::new(400.0, 400.0)), GizmoAxis::None);
    }

    #[test]
    fn no_handles_means_no_pick() {
        assert_eq!(pick(&[], Vec2::new(10.0, 10.0)), GizmoAxis::None);
    }

    #[test]
    fn the_nearest_handle_wins() {
        let handles = vec![
            ProjectedHandle {
                axis: GizmoAxis::X,
                origin: Vec2::new(0.0, 0.0),
                tip: Vec2::new(100.0, 0.0),
                pick_radius: 20.0,
            },
            ProjectedHandle {
                axis: GizmoAxis::Y,
                origin: Vec2::new(0.0, 0.0),
                tip: Vec2::new(0.0, 50.0),
                pick_radius: 20.0,
            },
        ];
        // Right on the X line, far from Y.
        assert_eq!(pick(&handles, Vec2::new(50.0, 1.0)), GizmoAxis::X);
        assert_eq!(pick(&handles, Vec2::new(1.0, 40.0)), GizmoAxis::Y);
    }

    #[test]
    fn a_tie_goes_to_the_earlier_axis() {
        // A cursor exactly on both handles must pick the same one every frame,
        // or a drag stutters between two directions as the mouse jitters.
        let handles = vec![
            ProjectedHandle {
                axis: GizmoAxis::X,
                origin: Vec2::new(0.0, 0.0),
                tip: Vec2::new(100.0, 0.0),
                pick_radius: 20.0,
            },
            ProjectedHandle {
                axis: GizmoAxis::Y,
                origin: Vec2::new(0.0, 0.0),
                tip: Vec2::new(100.0, 0.0),
                pick_radius: 20.0,
            },
        ];
        let first = pick(&handles, Vec2::new(50.0, 0.0));
        for _ in 0..10 {
            assert_eq!(pick(&handles, Vec2::new(50.0, 0.0)), first);
        }
        assert_eq!(first, GizmoAxis::X);
    }

    #[test]
    fn a_handle_contains_a_cursor_on_its_line() {
        let h = ProjectedHandle {
            axis: GizmoAxis::X,
            origin: Vec2::new(0.0, 0.0),
            tip: Vec2::new(100.0, 0.0),
            pick_radius: 6.0,
        };
        assert!(h.contains(Vec2::new(50.0, 0.0), 1.0));
        assert!(!h.contains(Vec2::new(50.0, 50.0), 1.0));
    }

    #[test]
    fn a_very_short_handle_is_still_pickable() {
        // A gizmo seen edge-on has a near-zero-length handle; a zero-length
        // segment must still measure to its point rather than dividing by zero.
        let d = distance_to_segment(Vec2::new(5.0, 0.0), Vec2::ZERO, Vec2::ZERO);
        assert!((d - 5.0).abs() < 1e-6, "{d}");
    }

    #[test]
    fn a_distance_beyond_the_segment_clamps_to_its_end() {
        let d = distance_to_segment(Vec2::new(150.0, 0.0), Vec2::ZERO, Vec2::new(100.0, 0.0));
        assert!((d - 50.0).abs() < 1e-6, "{d}");
    }

    #[test]
    fn the_closest_point_on_a_line_is_on_the_line() {
        let p = closest_point_on_line(Vec3::ZERO, Vec3::X, Vec3::new(5.0, 3.0, 0.0));
        assert!((p - Vec3::new(5.0, 0.0, 0.0)).length() < 1e-5, "{p:?}");
    }

    #[test]
    fn the_closest_point_handles_a_negative_direction() {
        let p = closest_point_on_line(Vec3::ZERO, Vec3::X, Vec3::new(5.0, 0.0, 0.0));
        assert!((p - Vec3::new(5.0, 0.0, 0.0)).length() < 1e-5, "{p:?}");
    }

    #[test]
    fn a_zero_direction_leaves_the_point_alone() {
        let p = closest_point_on_line(Vec3::ZERO, Vec3::ZERO, Vec3::X);
        assert_eq!(p, Vec3::ZERO, "a zero direction has no closest point");
    }

    #[test]
    fn an_unnormalised_direction_still_works() {
        // A gizmo's axis is a unit vector, but a caller may pass a scaled one;
        // the result should not depend on the scale.
        let p = closest_point_on_line(Vec3::ZERO, Vec3::X * 17.0, Vec3::new(5.0, 3.0, 0.0));
        assert!((p - Vec3::new(5.0, 0.0, 0.0)).length() < 1e-4, "{p:?}");
    }

    #[test]
    fn a_drag_along_an_axis_measures_that_axis() {
        let d = axis_delta(Vec3::ZERO, Vec3::ZERO, Vec3::X, Vec3::new(4.0, 9.0, 9.0));
        assert!((d - 4.0).abs() < 1e-5, "only the axis counts: {d}");
    }

    #[test]
    fn a_drag_against_the_axis_is_negative() {
        let d = axis_delta(Vec3::ZERO, Vec3::ZERO, Vec3::X, Vec3::new(-4.0, 0.0, 0.0));
        assert!((d + 4.0).abs() < 1e-5, "{d}");
    }

    #[test]
    fn a_drag_perpendicular_to_the_axis_measures_zero() {
        let d = axis_delta(Vec3::ZERO, Vec3::ZERO, Vec3::X, Vec3::new(0.0, 5.0, 0.0));
        assert!(d.abs() < 1e-5, "{d}");
    }

    #[test]
    fn a_drag_is_measured_from_its_start_not_accumulated() {
        // This is the property that stops a dropped frame from making the
        // entity drift: recomputing from the origin is idempotent.
        let once = axis_delta(Vec3::ZERO, Vec3::ZERO, Vec3::X, Vec3::new(3.0, 0.0, 0.0));
        let twice = axis_delta(Vec3::ZERO, Vec3::ZERO, Vec3::X, Vec3::new(3.0, 0.0, 0.0));
        assert_eq!(once, twice);
        assert!((once - 3.0).abs() < 1e-5);
    }

    #[test]
    fn a_drag_from_an_offset_origin_measures_the_same_distance() {
        let start = Vec3::new(10.0, 0.0, 0.0);
        let d = axis_delta(start, start, Vec3::X, start + Vec3::new(2.0, 0.0, 0.0));
        assert!((d - 2.0).abs() < 1e-5, "{d}");
    }

    #[test]
    fn a_drag_can_begin_on_each_axis() {
        for axis in GizmoAxis::singles() {
            let drag = GizmoDrag::begin(axis, Vec3::ZERO, Vec3::ZERO).unwrap();
            assert_eq!(drag.axis, axis);
            assert!(drag.is_active());
        }
    }

    #[test]
    fn a_drag_cannot_begin_on_a_screen_handle() {
        assert!(GizmoDrag::begin(GizmoAxis::Screen, Vec3::ZERO, Vec3::ZERO).is_none());
    }

    #[test]
    fn a_drag_cannot_begin_on_nothing() {
        assert!(GizmoDrag::begin(GizmoAxis::None, Vec3::ZERO, Vec3::ZERO).is_none());
    }

    #[test]
    fn a_drag_updates_its_delta() {
        let mut drag = GizmoDrag::begin(GizmoAxis::X, Vec3::ZERO, Vec3::ZERO).unwrap();
        let d = drag.update(Vec3::new(5.0, 0.0, 0.0));
        assert!((d - 5.0).abs() < 1e-5, "{d}");
        assert!((drag.delta - 5.0).abs() < 1e-5);
    }

    #[test]
    fn a_drag_back_to_its_start_returns_to_zero() {
        let mut drag = GizmoDrag::begin(GizmoAxis::X, Vec3::ZERO, Vec3::ZERO).unwrap();
        drag.update(Vec3::new(5.0, 0.0, 0.0));
        assert!(
            drag.update(Vec3::ZERO).abs() < 1e-5,
            "the origin is the reference"
        );
    }

    #[test]
    fn a_drag_on_no_axis_keeps_its_delta() {
        let mut drag = GizmoDrag {
            axis: GizmoAxis::None,
            start_origin: Vec3::ZERO,
            start_point: Vec3::ZERO,
            delta: 3.0,
        };
        assert_eq!(drag.update(Vec3::X * 10.0), 3.0);
        assert!(!drag.is_active());
    }

    #[test]
    fn a_world_space_gizmo_is_not_rotated() {
        let world = Mat4::from_rotation_z(std::f32::consts::FRAC_PI_2);
        assert_eq!(gizmo_rotation(world, false), Mat3::IDENTITY);
    }

    #[test]
    fn a_local_space_gizmo_takes_the_objects_rotation() {
        let world = Mat4::from_rotation_z(std::f32::consts::FRAC_PI_2);
        let r = gizmo_rotation(world, true);
        let y = r * Vec3::Y;
        assert!(
            (y - Vec3::new(-1.0, 0.0, 0.0)).length() < 1e-5,
            "a quarter turn takes +Y to -X: {y:?}"
        );
    }

    #[test]
    fn a_local_gizmo_stays_orthonormal() {
        let world = Mat4::from_scale_rotation_translation(
            Vec3::new(2.0, 2.0, 2.0),
            glam::Quat::from_rotation_x(0.7),
            Vec3::new(5.0, 0.0, 0.0),
        );
        let r = gizmo_rotation(world, true);
        let x = r * Vec3::X;
        let y = r * Vec3::Y;
        assert!(
            (x.length() - 1.0).abs() < 1e-5,
            "a scale must not leak in: {x:?}"
        );
        assert!(x.dot(y).abs() < 1e-5);
    }

    #[test]
    fn axes_have_indices_and_labels() {
        axes_have_indices();
    }

    /// A camera on the diagonal aimed at the origin, so a point in front of it
    /// is genuinely in front rather than off to one side.
    fn looking_at_origin() -> EditorCamera {
        let mut c = EditorCamera::perspective();
        c.position = Vec3::new(6.0, 5.0, 6.0);
        c.yaw = std::f32::consts::FRAC_PI_4;
        c.pitch = -0.6;
        c
    }

    fn viewport() -> Vec2 {
        Vec2::new(800.0, 600.0)
    }

    #[test]
    fn a_ray_points_away_from_the_camera() {
        let cam = looking_at_origin();
        let ray = screen_ray_with_camera(Vec2::new(400.0, 300.0), &cam, viewport())
            .expect("a ray through the middle of the viewport");
        // The camera looks at the origin, so the ray from the middle of the
        // screen must point back towards it. A ray pointing the other way is
        // what an un-undone Y flip or a wrong depth convention looks like.
        let to_origin = (Vec3::ZERO - cam.position).normalize_or_zero();
        assert!(
            ray.direction.dot(to_origin) > 0.9,
            "the ray points {:?}, away from the origin at {to_origin:?}",
            ray.direction
        );
    }

    #[test]
    fn a_ray_has_a_unit_direction() {
        let cam = looking_at_origin();
        let ray = screen_ray_with_camera(Vec2::new(120.0, 480.0), &cam, viewport()).expect("a ray");
        assert!(
            (ray.direction.length() - 1.0).abs() < 1.0e-4,
            "direction length was {}",
            ray.direction.length()
        );
    }

    #[test]
    fn a_ray_at_a_projected_point_passes_through_it() {
        // The round trip that matters: project a world point, unproject the
        // screen position, and the ray must pass through the original point.
        // Anything less and a resize handle would drift away from the cursor.
        let cam = looking_at_origin();
        let vp = viewport();
        let world = Vec3::new(0.5, -0.25, 0.0);
        let screen = project_with_camera(world, &cam, vp).expect("the point is in front");
        let ray = screen_ray_with_camera(screen, &cam, vp).expect("a ray");
        // The point is on the ray if the vector from the origin to it is
        // parallel to the direction.
        let to_point = world - ray.origin;
        assert!(
            to_point.length() > 0.1,
            "the point is on top of the camera: {world:?}"
        );
        let parallel = to_point.normalize_or_zero().dot(ray.direction).abs();
        assert!(
            parallel > 0.999,
            "the ray misses the point it came from: dir {:?}, offset {to_point:?}",
            ray.direction
        );
    }

    #[test]
    fn a_ray_through_the_centre_hits_a_box_at_the_origin() {
        let cam = looking_at_origin();
        let ray = screen_ray_with_camera(Vec2::new(400.0, 300.0), &cam, viewport()).expect("a ray");
        let hit = ray_hits_box(ray.origin, ray.direction, Vec3::ZERO, Vec3::splat(1.0))
            .expect("a box at the origin is in the middle of the view");
        // The near face of the box, so the hit is half a unit from the centre
        // along the ray.
        assert!(
            hit.distance(Vec3::ZERO) < 1.0,
            "the hit was {hit:?}, which is past the box's near face"
        );
    }

    #[test]
    fn a_ray_away_from_the_viewport_misses_the_box() {
        let cam = looking_at_origin();
        // The top-left corner looks well off to one side of a box at the origin.
        let ray = screen_ray_with_camera(Vec2::new(2.0, 2.0), &cam, viewport()).expect("a ray");
        assert!(
            ray_hits_box(ray.origin, ray.direction, Vec3::ZERO, Vec3::splat(1.0)).is_none(),
            "a ray through the corner should miss a unit box at the origin"
        );
    }

    #[test]
    fn a_ray_pointing_away_from_the_box_misses_it() {
        // The ray starts at the origin travelling +Z, and the box is at z = -10.
        // That is behind it in the only sense that matters: the span of the
        // crossing is wholly negative, so every plane is behind the ray.
        //
        // The direction is the part that is easy to get backwards. A ray at
        // z = 20 travelling -Z *does* hit a box at z = -10, because it reaches
        // it — the box is in front of that ray, however negative its coordinate.
        let hit = ray_hits_box(
            Vec3::ZERO,
            Vec3::Z,
            Vec3::new(0.0, 0.0, -10.0),
            Vec3::splat(1.0),
        );
        assert!(hit.is_none(), "a box behind the ray was hit at {hit:?}");
    }

    #[test]
    fn a_ray_starting_inside_a_box_reports_the_exit() {
        // A camera inside a box is unusual but not impossible, and the slab test
        // must not divide by zero or return nonsense. The reported point is the
        // far face, which is the only surface there is to see from in there.
        let hit = ray_hits_box(
            Vec3::ZERO,
            Vec3::Z,
            Vec3::new(0.0, 0.0, 5.0),
            Vec3::splat(2.0),
        )
        .expect("a ray starting inside a box still has an exit");
        assert!(
            (hit.z - 4.0).abs() < 1.0e-4,
            "expected the far face at z=4, got {hit:?}"
        );
    }

    #[test]
    fn a_ray_parallel_to_a_slab_outside_it_misses_the_box() {
        // Travelling along X at a Y and Z outside the box: the X slab never
        // constrains, and the other two say no.
        let hit = ray_hits_box(
            Vec3::new(-10.0, 5.0, 5.0),
            Vec3::X,
            Vec3::ZERO,
            Vec3::splat(1.0),
        );
        assert!(hit.is_none());
    }

    #[test]
    fn a_ray_parallel_to_a_slab_inside_it_can_hit() {
        // The same ray inside the box's Y and Z range does hit, on the -X face.
        let hit = ray_hits_box(
            Vec3::new(-10.0, 0.0, 0.0),
            Vec3::X,
            Vec3::ZERO,
            Vec3::splat(2.0),
        )
        .expect("a ray along the box's own axis and inside it must hit");
        assert!(
            (hit.x + 1.0).abs() < 1.0e-4,
            "hit the wrong face at {hit:?}"
        );
    }

    #[test]
    fn a_larger_box_is_hit_where_a_smaller_one_is_not() {
        // The discriminating test for a slab test that is not actually testing
        // all three axes: a ray that grazes a small box but passes through a
        // large one.
        let origin = Vec3::new(0.0, 0.0, 5.0);
        let dir = Vec3::new(0.0, 0.0, -1.0);
        assert!(ray_hits_box(origin, dir, Vec3::ZERO, Vec3::splat(1.0)).is_some());
        // A box offset in X that this ray, straight down -Z at x=0, cannot reach.
        assert!(
            ray_hits_box(origin, dir, Vec3::new(20.0, 0.0, 0.0), Vec3::splat(1.0)).is_none(),
            "a ray straight down -Z at x=0 hit a box centred at x=20"
        );
    }

    #[test]
    fn a_ray_meets_a_plane_it_points_at() {
        let ray = Ray {
            origin: Vec3::ZERO,
            direction: Vec3::Z,
        };
        let hit = ray
            .intersect_plane(Vec3::new(0.0, 0.0, 5.0), Vec3::Z)
            .expect("a ray pointing at the plane meets it");
        assert!((hit.z - 5.0).abs() < 1.0e-4, "met it at {hit:?}");
    }

    #[test]
    fn a_ray_parallel_to_a_plane_does_not_meet_it() {
        // The case that would otherwise divide by zero and put the handle at
        // infinity.
        let ray = Ray {
            origin: Vec3::ZERO,
            direction: Vec3::X,
        };
        assert!(
            ray.intersect_plane(Vec3::new(0.0, 0.0, 5.0), Vec3::Z)
                .is_none()
        );
    }

    #[test]
    fn a_ray_pointing_away_from_a_plane_does_not_meet_it() {
        // Behind the ray, not in front of it: the drag has no answer, and the box
        // should stay put rather than jump.
        let ray = Ray {
            origin: Vec3::ZERO,
            direction: Vec3::NEG_Z,
        };
        assert!(
            ray.intersect_plane(Vec3::new(0.0, 0.0, 5.0), Vec3::Z)
                .is_none()
        );
    }

    #[test]
    fn a_zero_sized_viewport_produces_no_ray() {
        // A minimised window reports a zero extent, and a ray through it would
        // be a division by zero.
        let cam = looking_at_origin();
        assert!(screen_ray_with_camera(Vec2::new(10.0, 10.0), &cam, Vec2::ZERO).is_none());
    }
}
