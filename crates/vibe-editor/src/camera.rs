//! Editor cameras: a free-fly 3D camera and a screen-space 2D camera.
//!
//! Both types are plain data with `f32` clamps rather than error types. A
//! slider dragged past its limit is not a failure a caller can recover from, so
//! the cameras absorb the bad value and stay in a drawable state instead of
//! propagating a `Result` through every egui panel.

use glam::Mat4;
use glam::Vec2;
use glam::Vec3;
use glam::Vec4;
use glam::camera::rh::proj::vulkan;
use glam::camera::rh::view::look_to_mat4;

/// The default vertical field of view: 60 degrees.
pub const DEFAULT_FOV_Y_RADIANS: f32 = std::f32::consts::FRAC_PI_3;

/// The default near plane distance.
pub const DEFAULT_NEAR: f32 = 0.1;

/// The default far plane distance.
pub const DEFAULT_FAR: f32 = 1000.0;

/// The default half-height of the orthographic frustum.
pub const DEFAULT_ORTHOGRAPHIC_SIZE: f32 = 10.0;

/// The default width-over-height of the surface being drawn to.
pub const DEFAULT_ASPECT: f32 = 1.0;

/// The default scale of a 2D camera, in pixels per world unit.
pub const DEFAULT_2D_PIXELS_PER_UNIT: f32 = 32.0;

/// The smallest orthographic half-height, in world units.
pub const MIN_ORTHOGRAPHIC_SIZE: f32 = 1.0e-3;

/// The smallest 2D scale, in pixels per world unit.
pub const MIN_PIXELS_PER_UNIT: f32 = 1.0e-3;

/// How far the pitch is held back from straight up or straight down, in radians.
pub const PITCH_EPSILON: f32 = 1.0e-3;

/// The largest pitch an editor camera can reach, in radians.
pub const PITCH_LIMIT: f32 = std::f32::consts::FRAC_PI_2 - PITCH_EPSILON;

const MIN_NEAR: f32 = 1.0e-4;
const MIN_FOV_Y_RADIANS: f32 = 1.0e-3;

/// A free-fly editor camera, perspective or orthographic.
///
/// Position, yaw and pitch are world space: yaw of zero looks down `-Z`, and
/// positive pitch looks up. Movement, orbit and zoom all take per-frame deltas
/// in world units or radians, so a panel feeds them raw egui drag deltas
/// scaled by its own sensitivity.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EditorCamera {
    /// Eye position in world space.
    pub position: Vec3,
    /// Rotation about the world up axis, in radians.
    pub yaw: f32,
    /// Rotation about the camera's right axis, in radians.
    pub pitch: f32,
    /// Vertical field of view, in radians; perspective only.
    pub fov_y_radians: f32,
    /// Near plane distance from the eye.
    pub near: f32,
    /// Far plane distance from the eye.
    pub far: f32,
    /// Half-height of the orthographic frustum; orthographic only.
    pub orthographic_size: f32,
    /// Width over height of the surface being drawn to.
    pub aspect: f32,
    /// Whether [`EditorCamera::projection`] builds an orthographic matrix.
    pub projection_is_orthographic: bool,
}

impl EditorCamera {
    /// A perspective camera at `(0, 0, 5)` looking down `-Z`.
    pub fn perspective() -> Self {
        Self {
            position: Vec3::new(0.0, 0.0, 5.0),
            yaw: 0.0,
            pitch: 0.0,
            fov_y_radians: DEFAULT_FOV_Y_RADIANS,
            near: DEFAULT_NEAR,
            far: DEFAULT_FAR,
            orthographic_size: DEFAULT_ORTHOGRAPHIC_SIZE,
            aspect: DEFAULT_ASPECT,
            projection_is_orthographic: false,
        }
    }

    /// An orthographic camera at `(0, 0, 5)` looking down `-Z`.
    pub fn orthographic() -> Self {
        Self {
            projection_is_orthographic: true,
            ..Self::perspective()
        }
    }

    /// The horizontal forward axis, ignoring pitch.
    ///
    /// Deliberately pitch-free: fly-to-camera controls and middle-drag panning
    /// both want a ground-relative forward, and a pitch-aware forward would
    /// drive the eye into the floor as the user tilts down.
    pub fn forward(&self) -> Vec3 {
        Vec3::new(-self.yaw.sin(), 0.0, -self.yaw.cos())
    }

    /// The right axis, ignoring pitch.
    pub fn right(&self) -> Vec3 {
        Vec3::new(self.yaw.cos(), 0.0, -self.yaw.sin())
    }

    /// The up axis, ignoring pitch.
    pub fn up(&self) -> Vec3 {
        self.right().cross(self.forward())
    }

    /// The direction the camera is actually looking, pitch included.
    ///
    /// This is what [`EditorCamera::view`] points at, and it differs from
    /// [`EditorCamera::forward`] whenever the pitch is non-zero.
    pub fn look_direction(&self) -> Vec3 {
        let (sin_yaw, cos_yaw) = self.yaw.sin_cos();
        let (sin_pitch, cos_pitch) = self.pitch.sin_cos();
        Vec3::new(-sin_yaw * cos_pitch, sin_pitch, -cos_yaw * cos_pitch)
    }

    /// The right-handed view matrix, looking down the camera's own -Z axis.
    pub fn view(&self) -> Mat4 {
        // look_to takes a direction and asserts that it is already normalised,
        // unlike look_at which normalises a difference for you.
        look_to_mat4(self.position, self.look_direction().normalize(), self.up())
    }

    /// The projection matrix for a surface of the given width over height.
    ///
    /// A non-finite or non-positive `aspect` falls back to the stored
    /// [`EditorCamera::aspect`], because a minimised window reports a zero
    /// aspect and a `NaN` matrix would silently blank every frame.
    pub fn projection(&self, aspect: f32) -> Mat4 {
        let aspect = if aspect.is_finite() && aspect > 0.0 {
            aspect
        } else {
            self.aspect
        };
        // glam asserts on a non-positive near plane, and a zero field of view
        // divides by a zero sine, so neither is left to the caller.
        let near = self.near.max(MIN_NEAR);
        let far = self.far.max(near * 2.0);
        if self.projection_is_orthographic {
            let half_height = self.orthographic_size.max(MIN_ORTHOGRAPHIC_SIZE);
            let half_width = half_height * aspect;
            orthographic(
                -half_width,
                half_width,
                -half_height,
                half_height,
                near,
                far,
            )
        } else {
            let fov = self
                .fov_y_radians
                .clamp(MIN_FOV_Y_RADIANS, std::f32::consts::PI - MIN_FOV_Y_RADIANS);
            vulkan::perspective(fov, aspect, near, far)
        }
    }

    /// Projection times view, ready to upload as a uniform.
    pub fn view_projection(&self, aspect: f32) -> Mat4 {
        self.projection(aspect) * self.view()
    }

    /// Record the surface aspect, ignoring a non-finite or non-positive value.
    pub fn set_aspect(&mut self, aspect: f32) {
        if aspect.is_finite() && aspect > 0.0 {
            self.aspect = aspect;
        }
    }

    /// Translate along [`EditorCamera::forward`].
    pub fn move_forward(&mut self, amount: f32) {
        self.position += self.forward() * finite_or_zero(amount);
    }

    /// Translate along [`EditorCamera::right`].
    pub fn move_right(&mut self, amount: f32) {
        self.position += self.right() * finite_or_zero(amount);
    }

    /// Translate along [`EditorCamera::up`].
    pub fn move_up(&mut self, amount: f32) {
        self.position += self.up() * finite_or_zero(amount);
    }

    /// Rotate by a yaw and pitch delta, in radians.
    ///
    /// The pitch is clamped to [`PITCH_LIMIT`] so the camera cannot roll over
    /// the poles. The yaw accumulates without wrapping; `sin_cos` stays
    /// accurate long before a session's worth of dragging loses precision.
    pub fn orbit(&mut self, delta_yaw: f32, delta_pitch: f32) {
        self.yaw += finite_or_zero(delta_yaw);
        self.pitch = (self.pitch + finite_or_zero(delta_pitch)).clamp(-PITCH_LIMIT, PITCH_LIMIT);
    }

    /// Zoom by a scroll delta; positive moves closer to what the camera sees.
    ///
    /// A perspective camera dollies along [`EditorCamera::forward`], an
    /// orthographic one scales [`EditorCamera::orthographic_size`], which is
    /// held above [`MIN_ORTHOGRAPHIC_SIZE`] because a zero size makes the
    /// projection singular.
    pub fn zoom(&mut self, amount: f32) {
        let amount = finite_or_zero(amount);
        if self.projection_is_orthographic {
            self.orthographic_size =
                (self.orthographic_size * (1.0 + amount)).max(MIN_ORTHOGRAPHIC_SIZE);
        } else {
            self.position += self.forward() * amount;
        }
    }
}

/// Per-frame camera deltas, filled in by the egui panels and applied in one go.
///
/// The fields are plain `f32` so a panel can accumulate a drag into them
/// without allocating; units are world units, radians or scroll notches, all
/// already scaled by the panel's own sensitivity.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct CameraInput {
    /// Forward translation, in world units.
    pub move_forward: f32,
    /// Right translation, in world units.
    pub move_right: f32,
    /// Up translation, in world units.
    pub move_up: f32,
    /// Yaw delta, in radians.
    pub orbit_yaw: f32,
    /// Pitch delta, in radians.
    pub orbit_pitch: f32,
    /// Scroll notches; positive zooms in.
    pub zoom: f32,
    /// Right pan, in world units.
    pub pan_right: f32,
    /// Up pan, in world units.
    pub pan_up: f32,
}

impl CameraInput {
    /// No motion at all.
    pub fn zeroed() -> Self {
        Self::default()
    }

    /// Apply every delta to `camera`.
    ///
    /// Movement comes first, then orbit, then zoom, then pan, so a drag that
    /// also orbits leaves the camera at the position the cursor was dragged
    /// from. Pan duplicates `move_*` only so a panel can scale a pixel drag
    /// independently of a key press; both translate along the same axes.
    pub fn apply(&self, camera: &mut EditorCamera) {
        camera.move_forward(self.move_forward);
        camera.move_right(self.move_right);
        camera.move_up(self.move_up);
        camera.orbit(self.orbit_yaw, self.orbit_pitch);
        camera.zoom(self.zoom);
        camera.move_right(self.pan_right);
        camera.move_up(self.pan_up);
    }
}

/// A screen-space orthographic camera for 2D viewports.
///
/// The world is measured in whatever units the caller likes and the camera
/// tracks how many pixels one of them covers, so panning is in world units and
/// zooming is a single scale factor.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Camera2DController {
    center: Vec2,
    pixels_per_unit: f32,
}

impl Camera2DController {
    /// A camera centred on `center` with `pixels_per_unit` pixels per world unit.
    ///
    /// A non-positive scale is floored at [`MIN_PIXELS_PER_UNIT`] and a
    /// non-finite one falls back to [`DEFAULT_2D_PIXELS_PER_UNIT`], since
    /// either would make the projection singular.
    pub fn new(center: Vec2, pixels_per_unit: f32) -> Self {
        Self {
            center,
            pixels_per_unit: sanitize_scale(pixels_per_unit),
        }
    }

    /// The world point at the middle of the view.
    pub fn center(&self) -> Vec2 {
        self.center
    }

    /// Pixels per world unit; always at least [`MIN_PIXELS_PER_UNIT`].
    pub fn pixels_per_unit(&self) -> f32 {
        self.pixels_per_unit
    }

    /// Slide the view by a world-space delta, exactly.
    pub fn pan(&mut self, dx: f32, dy: f32) {
        self.center += Vec2::new(finite_or_zero(dx), finite_or_zero(dy));
    }

    /// Zoom by a signed scroll amount, returning the factor that was applied.
    ///
    /// The factor is `1.0 + amount` once no clamp bites, and `1.0` when the
    /// scale is already resting on [`MIN_PIXELS_PER_UNIT`] and a zoom-out
    /// asked for more. A caller anchoring a pan to the cursor chains the
    /// returned factor onto the drag rather than re-deriving a clamped value.
    pub fn zoom_at(&mut self, amount: f32) -> f32 {
        let before = self.pixels_per_unit;
        self.pixels_per_unit = sanitize_scale(before * (1.0 + finite_or_zero(amount)));
        self.pixels_per_unit / before
    }

    /// The world-to-clip matrix for a `viewport` in pixels, with Y up.
    ///
    /// A degenerate viewport (a minimised panel reports zero or negative size)
    /// is floored at one pixel so the matrix stays finite.
    pub fn view_projection(&self, viewport: Vec2) -> Mat4 {
        let half = Vec2::new(sanitise_viewport(viewport.x), sanitise_viewport(viewport.y))
            * (0.5 / self.pixels_per_unit);
        orthographic(
            self.center.x - half.x,
            self.center.x + half.x,
            self.center.y - half.y,
            self.center.y + half.y,
            0.0,
            1.0,
        )
    }
}

/// A right-handed orthographic projection with a zero-to-one depth range.
///
/// Hand-built rather than taken from glam's `vulkan::orthographic`, which
/// applies a Y-flip: its `y_axis.y` comes out negative, so world up lands at
/// the bottom of the frame and the 2D camera renders upside down. Depth here
/// runs along the view space's -Z axis, so a point in front of the camera
/// sits at negative Z and a zero-to-one range spans z from 0 to -far.
fn orthographic(left: f32, right: f32, bottom: f32, top: f32, near: f32, far: f32) -> Mat4 {
    Mat4::from_cols(
        Vec4::new(2.0 / (right - left), 0.0, 0.0, 0.0),
        Vec4::new(0.0, 2.0 / (top - bottom), 0.0, 0.0),
        Vec4::new(0.0, 0.0, -1.0 / (far - near), 0.0),
        Vec4::new(
            -(right + left) / (right - left),
            -(top + bottom) / (top - bottom),
            -near / (far - near),
            1.0,
        ),
    )
}

fn finite_or_zero(value: f32) -> f32 {
    if value.is_finite() { value } else { 0.0 }
}

fn sanitize_scale(value: f32) -> f32 {
    if value.is_nan() {
        DEFAULT_2D_PIXELS_PER_UNIT
    } else {
        value.clamp(MIN_PIXELS_PER_UNIT, f32::MAX)
    }
}

fn sanitise_viewport(value: f32) -> f32 {
    if value.is_finite() {
        value.max(1.0)
    } else {
        1.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ASPECTS: [f32; 4] = [0.1, 1.0, 16.0 / 9.0, 3.0];

    fn assert_close(got: f32, want: f32) {
        assert!((got - want).abs() < 1e-5, "got {got}, want {want}");
    }

    fn assert_vec_close(got: Vec3, want: Vec3) {
        assert!((got - want).length() < 1e-5, "got {got:?}, want {want:?}");
    }

    fn assert_finite(m: Mat4) {
        assert!(m.is_finite(), "matrix is not finite: {m:?}");
    }

    fn ndc(matrix: Mat4, point: Vec3) -> Vec3 {
        let clip = matrix * point.extend(1.0);
        clip.truncate() / clip.w
    }

    fn ndc2(matrix: Mat4, point: Vec2) -> Vec3 {
        ndc(matrix, point.extend(0.0))
    }

    fn looking(yaw: f32, pitch: f32) -> EditorCamera {
        EditorCamera {
            yaw,
            pitch,
            ..EditorCamera::perspective()
        }
    }

    #[test]
    fn perspective_defaults_are_the_documented_ones() {
        let cam = EditorCamera::perspective();
        assert_eq!(cam.position, Vec3::new(0.0, 0.0, 5.0));
        assert_close(cam.yaw, 0.0);
        assert_close(cam.pitch, 0.0);
        assert_close(cam.fov_y_radians, std::f32::consts::FRAC_PI_3);
        assert_close(cam.near, 0.1);
        assert_close(cam.far, 1000.0);
        assert_close(cam.orthographic_size, 10.0);
        assert_close(cam.aspect, 1.0);
        assert!(!cam.projection_is_orthographic);
    }

    #[test]
    fn orthographic_defaults_match_perspective_except_the_flag() {
        let ortho = EditorCamera::orthographic();
        let persp = EditorCamera::perspective();
        assert!(ortho.projection_is_orthographic);
        assert_eq!(ortho.position, persp.position);
        assert_close(ortho.fov_y_radians, persp.fov_y_radians);
        assert_close(ortho.near, persp.near);
        assert_close(ortho.far, persp.far);
        assert_close(ortho.orthographic_size, persp.orthographic_size);
        assert_eq!(
            ortho,
            EditorCamera {
                projection_is_orthographic: true,
                ..persp
            }
        );
    }

    #[test]
    fn zero_yaw_looks_down_negative_z() {
        let cam = EditorCamera::perspective();
        assert_vec_close(cam.forward(), Vec3::NEG_Z);
        assert_vec_close(cam.look_direction(), Vec3::NEG_Z);
        assert_vec_close(cam.right(), Vec3::X);
        assert_vec_close(cam.up(), Vec3::Y);
    }

    #[test]
    fn forward_right_and_up_ignore_pitch() {
        for pitch in [-1.4, -0.3, 0.0, 0.9, 1.55] {
            let cam = looking(0.6, pitch);
            assert_vec_close(cam.forward(), looking(0.6, 0.0).forward());
            assert_vec_close(cam.right(), looking(0.6, 0.0).right());
            assert_vec_close(cam.up(), Vec3::Y);
        }
    }

    #[test]
    fn look_direction_follows_the_pitch() {
        let up = looking(0.0, PITCH_LIMIT).look_direction();
        assert!(up.y > 0.99, "{up:?}");
        let down = looking(0.0, -PITCH_LIMIT).look_direction();
        assert!(down.y < -0.99, "{down:?}");
    }

    #[test]
    fn axes_are_unit_length_and_perpendicular() {
        for yaw in [0.0, 0.4, 1.0, -2.1, 3.0, 6.0] {
            for pitch in [0.0, 0.3, -1.4, 1.55, -0.9] {
                let cam = looking(yaw, pitch);
                let (f, r, u) = (cam.forward(), cam.right(), cam.up());
                assert_close(f.length(), 1.0);
                assert_close(r.length(), 1.0);
                assert_close(u.length(), 1.0);
                assert_close(f.dot(r), 0.0);
                assert_close(f.dot(u), 0.0);
                assert_close(r.dot(u), 0.0);
            }
        }
    }

    #[test]
    fn axes_form_a_right_handed_frame() {
        let cam = looking(1.1, 0.4);
        assert_vec_close(cam.right().cross(cam.forward()), cam.up());
    }

    #[test]
    fn look_direction_is_unit_for_any_orientation() {
        for yaw in [-3.0, -0.2, 0.0, 1.7] {
            for pitch in [-PITCH_LIMIT, -0.5, 0.0, 0.5, PITCH_LIMIT] {
                let cam = looking(yaw, pitch);
                assert_close(cam.look_direction().length(), 1.0);
            }
        }
    }

    #[test]
    fn view_puts_the_eye_at_the_view_origin() {
        let cam = EditorCamera::perspective();
        let in_view = cam.view() * cam.position.extend(1.0);
        assert!(
            in_view.x.abs() < 1e-5 && in_view.y.abs() < 1e-5,
            "{in_view:?}"
        );
    }

    #[test]
    fn view_maps_a_point_in_front_to_negative_z() {
        let cam = EditorCamera::perspective();
        let in_view = cam.view() * Vec3::ZERO.extend(1.0);
        assert!(in_view.z < 0.0, "{in_view:?}");
    }

    #[test]
    fn view_points_along_the_look_direction() {
        for yaw in [0.0, 0.9, -2.4] {
            for pitch in [-1.0, 0.0, 0.7] {
                let cam = looking(yaw, pitch);
                let target = cam.position + cam.look_direction();
                let in_view = cam.view() * target.extend(1.0);
                assert!(in_view.z < 0.0, "{in_view:?}");
                assert!(in_view.x.abs() < 1e-4, "{in_view:?}");
                assert!(in_view.y.abs() < 1e-4, "{in_view:?}");
            }
        }
    }

    #[test]
    fn view_is_finite_at_the_pitch_limits() {
        for pitch in [-PITCH_LIMIT, 0.0, PITCH_LIMIT] {
            assert_finite(looking(1.3, pitch).view());
        }
    }

    #[test]
    fn perspective_projection_maps_near_to_zero_and_far_to_one() {
        let cam = EditorCamera::perspective();
        let proj = cam.projection(1.0);
        let near = ndc(proj, Vec3::new(0.0, 0.0, -cam.near));
        let far = ndc(proj, Vec3::new(0.0, 0.0, -cam.far));
        assert_close(near.z, 0.0);
        assert_close(far.z, 1.0);
    }

    #[test]
    fn perspective_depth_increases_monotonically() {
        let proj = EditorCamera::perspective().projection(16.0 / 9.0);
        let mut previous = f32::NEG_INFINITY;
        for step in 0..64 {
            let distance = 0.5 + step as f32 * 15.0;
            let depth = ndc(proj, Vec3::new(0.3, -0.2, -distance)).z;
            assert!(depth > previous, "step {step}: {depth} <= {previous}");
            assert!((0.0..=1.0).contains(&depth), "step {step}: {depth}");
            previous = depth;
        }
    }

    #[test]
    fn perspective_projection_is_vulkan_y_down() {
        let proj = EditorCamera::perspective().projection(1.0);
        let above = ndc(proj, Vec3::new(0.0, 1.0, -2.0));
        assert!(above.y < 0.0, "Vulkan NDC is Y-down, got {above:?}");
    }

    #[test]
    fn perspective_projection_widens_with_aspect() {
        let cam = EditorCamera::perspective();
        let narrow = cam.projection(1.0);
        let wide = cam.projection(2.0);
        assert!(wide.x_axis.x < narrow.x_axis.x);
        assert_close(wide.y_axis.y, narrow.y_axis.y);
    }

    #[test]
    fn perspective_projection_survives_a_degenerate_frustum() {
        let mut cam = EditorCamera::perspective();
        cam.near = 0.0;
        cam.far = 0.0;
        cam.fov_y_radians = 0.0;
        assert_finite(cam.projection(1.0));
    }

    #[test]
    fn orthographic_projection_has_the_expected_terms() {
        let proj = EditorCamera::orthographic().projection(2.0);
        let (half_height, near, far) = (10.0f32, 0.1f32, 1000.0f32);
        let half_width = half_height * 2.0;
        assert_close(proj.x_axis.x, 2.0 / (2.0 * half_width));
        assert_close(proj.y_axis.y, 2.0 / (2.0 * half_height));
        assert_close(proj.z_axis.z, -1.0 / (far - near));
        assert_close(proj.w_axis.z, -near / (far - near));
        assert_close(proj.w_axis.w, 1.0);
    }

    #[test]
    fn orthographic_projection_centres_on_the_eye_axis() {
        let centre = ndc(
            EditorCamera::orthographic().projection(1.5),
            Vec3::new(0.0, 0.0, -3.0),
        );
        assert!(centre.x.abs() < 1e-5, "{centre:?}");
        assert!(centre.y.abs() < 1e-5, "{centre:?}");
    }

    #[test]
    fn orthographic_projection_is_y_up() {
        let proj = EditorCamera::orthographic().projection(1.0);
        let above = ndc(proj, Vec3::new(0.0, 1.0, -3.0));
        let right = ndc(proj, Vec3::new(1.0, 0.0, -3.0));
        assert!(above.y > 0.0, "{above:?}");
        assert!(right.x > 0.0, "{right:?}");
    }

    #[test]
    fn orthographic_projection_maps_near_and_far_to_the_unit_range() {
        let cam = EditorCamera::orthographic();
        let proj = cam.projection(1.0);
        let near = ndc(proj, Vec3::new(0.0, 0.0, -cam.near));
        let far = ndc(proj, Vec3::new(0.0, 0.0, -cam.far));
        assert_close(near.z, 0.0);
        assert_close(far.z, 1.0);
    }

    #[test]
    fn orthographic_projection_is_the_opposite_of_glam_vulkan_orthographic() {
        let ours = orthographic(-2.0, 2.0, -1.0, 1.0, 0.0, 10.0);
        let theirs = vulkan::orthographic(-2.0, 2.0, -1.0, 1.0, 0.0, 10.0);
        assert_close(ours.y_axis.y, -theirs.y_axis.y);
        assert!(ours.y_axis.y > 0.0 && theirs.y_axis.y < 0.0);
        assert_close(ours.x_axis.x, theirs.x_axis.x);
        assert_close(ours.z_axis.z, theirs.z_axis.z);
        assert_close(ours.w_axis.z, theirs.w_axis.z);
    }

    #[test]
    fn orthographic_projection_never_collapses_to_zero() {
        let mut cam = EditorCamera::orthographic();
        cam.orthographic_size = 0.0;
        let proj = cam.projection(1.0);
        assert!(proj.x_axis.x.is_finite() && proj.x_axis.x != 0.0);
        assert!(proj.y_axis.y.is_finite() && proj.y_axis.y != 0.0);
    }

    #[test]
    fn view_projection_equals_projection_times_view() {
        let cam = EditorCamera::perspective();
        let combined = cam.view_projection(16.0 / 9.0);
        let separate = cam.projection(16.0 / 9.0) * cam.view();
        for (got, want) in combined
            .to_cols_array()
            .iter()
            .zip(separate.to_cols_array())
        {
            assert!((got - want).abs() < 1e-5, "{got} != {want}");
        }
    }

    #[test]
    fn view_projection_is_finite_for_every_aspect() {
        for cam in [EditorCamera::perspective(), EditorCamera::orthographic()] {
            for aspect in ASPECTS {
                assert_finite(cam.projection(aspect));
                assert_finite(cam.view_projection(aspect));
            }
        }
    }

    #[test]
    fn view_projection_falls_back_when_the_aspect_is_invalid() {
        let mut cam = EditorCamera::perspective();
        cam.set_aspect(2.0);
        for aspect in [0.0, -1.0, f32::NAN, f32::INFINITY] {
            assert_finite(cam.view_projection(aspect));
            assert_eq!(cam.view_projection(aspect), cam.view_projection(2.0));
        }
    }

    #[test]
    fn view_projection_puts_a_target_in_front_inside_the_frustum() {
        let cam = EditorCamera::perspective();
        let target = cam.position + cam.look_direction() * 5.0;
        let clip = ndc(cam.view_projection(1.0), target);
        assert!(clip.x.abs() < 1e-4, "{clip:?}");
        assert!(clip.y.abs() < 1e-4, "{clip:?}");
        assert!((0.0..1.0).contains(&clip.z), "{clip:?}");
    }

    #[test]
    fn set_aspect_keeps_the_old_value_for_junk() {
        let mut cam = EditorCamera::perspective();
        for bad in [0.0, -1.0, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            cam.set_aspect(bad);
            assert_close(cam.aspect, DEFAULT_ASPECT);
        }
        cam.set_aspect(1.5);
        assert_close(cam.aspect, 1.5);
    }

    #[test]
    fn orbit_adds_the_yaw_delta() {
        let mut cam = EditorCamera::perspective();
        cam.orbit(0.25, 0.0);
        cam.orbit(0.25, 0.0);
        assert_close(cam.yaw, 0.5);
        assert_close(cam.pitch, 0.0);
    }

    #[test]
    fn orbit_clamps_the_pitch_at_the_limits() {
        let mut cam = EditorCamera::perspective();
        for _ in 0..1000 {
            cam.orbit(0.0, 10.0);
        }
        assert!(cam.pitch < std::f32::consts::FRAC_PI_2);
        assert_close(cam.pitch, PITCH_LIMIT);
        for _ in 0..1000 {
            cam.orbit(0.0, -10.0);
        }
        assert!(cam.pitch > -std::f32::consts::FRAC_PI_2);
        assert_close(cam.pitch, -PITCH_LIMIT);
    }

    #[test]
    fn orbit_keeps_the_pitch_inside_the_limits_for_a_mixed_sweep() {
        let mut cam = EditorCamera::perspective();
        for step in 0..500 {
            for delta in [0.7, -3.0, 0.1, -0.2, 12.0, -12.0, 0.0] {
                cam.orbit(0.13, delta);
                assert!(
                    cam.pitch < std::f32::consts::FRAC_PI_2
                        && cam.pitch > -std::f32::consts::FRAC_PI_2,
                    "step {step}: {}",
                    cam.pitch
                );
                assert_finite(cam.view());
            }
        }
    }

    #[test]
    fn orbit_ignores_a_non_finite_delta() {
        let mut cam = looking(0.3, 0.2);
        let before = cam;
        cam.orbit(f32::NAN, f32::INFINITY);
        assert_eq!(cam, before);
    }

    #[test]
    fn move_forward_travels_one_unit_along_forward() {
        let mut cam = looking(0.7, 0.9);
        let start = cam.position;
        let forward = cam.forward();
        cam.move_forward(1.0);
        assert_vec_close(cam.position, start + forward);
    }

    #[test]
    fn move_right_and_up_follow_their_axes() {
        let mut cam = looking(-0.4, 0.0);
        let start = cam.position;
        cam.move_right(2.0);
        assert_vec_close(cam.position, start + 2.0 * cam.right());
        cam.move_up(-3.0);
        assert_vec_close(cam.position, start + 2.0 * cam.right() - 3.0 * cam.up());
    }

    #[test]
    fn moving_forward_does_not_change_height_under_a_steep_pitch() {
        for pitch in [-1.5, -1.0, 1.0, 1.5] {
            let mut cam = looking(0.5, pitch);
            let start = cam.position;
            cam.move_forward(10.0);
            assert_close(cam.position.y, start.y);
            assert_vec_close(cam.position, start + 10.0 * cam.forward());
        }
    }

    #[test]
    fn movement_ignores_a_non_finite_amount() {
        let mut cam = looking(0.2, 0.1);
        let start = cam.position;
        cam.move_forward(f32::NAN);
        cam.move_right(f32::NEG_INFINITY);
        cam.move_up(f32::INFINITY);
        assert_eq!(cam.position, start);
    }

    #[test]
    fn perspective_zoom_dollies_along_forward() {
        let mut cam = EditorCamera::perspective();
        let start = cam.position;
        cam.zoom(1.0);
        assert_vec_close(cam.position, start + cam.forward());
        assert_close((cam.position - start).length(), 1.0);
        cam.zoom(-1.0);
        assert_vec_close(cam.position, start);
    }

    #[test]
    fn perspective_zoom_stays_finite_over_many_calls() {
        let mut cam = EditorCamera::perspective();
        for _ in 0..20_000 {
            cam.zoom(0.01);
        }
        assert!(cam.position.is_finite(), "{:?}", cam.position);
        assert_finite(cam.view_projection(16.0 / 9.0));
        for _ in 0..40_000 {
            cam.zoom(-0.01);
        }
        assert!(cam.position.is_finite(), "{:?}", cam.position);
        assert_finite(cam.view_projection(1.0));
    }

    #[test]
    fn perspective_zoom_ignores_a_non_finite_amount() {
        let mut cam = EditorCamera::perspective();
        let before = cam;
        cam.zoom(f32::NAN);
        assert_eq!(cam, before);
    }

    #[test]
    fn orthographic_zoom_scales_the_frustum() {
        let mut cam = EditorCamera::orthographic();
        let start = cam.orthographic_size;
        cam.zoom(0.5);
        assert_close(cam.orthographic_size, start * 1.5);
        cam.zoom(-0.5);
        assert_close(cam.orthographic_size, start * 1.5 * 0.5);
        assert_eq!(cam.position, EditorCamera::orthographic().position);
    }

    #[test]
    fn orthographic_zoom_never_reaches_zero() {
        let mut cam = EditorCamera::orthographic();
        for _ in 0..10_000 {
            cam.zoom(-0.5);
        }
        assert_close(cam.orthographic_size, MIN_ORTHOGRAPHIC_SIZE);
        assert!(cam.orthographic_size > 0.0);
        assert_finite(cam.view_projection(1.0));
        for _ in 0..10_000 {
            cam.zoom(-10.0);
        }
        assert_close(cam.orthographic_size, MIN_ORTHOGRAPHIC_SIZE);
        assert_finite(cam.projection(1.0));
    }

    #[test]
    fn orthographic_zoom_grows_without_bound_upwards() {
        let mut cam = EditorCamera::orthographic();
        for _ in 0..100 {
            cam.zoom(0.1);
        }
        assert!(cam.orthographic_size > 1.0, "{}", cam.orthographic_size);
        assert_finite(cam.projection(1.0));
    }

    #[test]
    fn camera_input_zeroed_is_all_zero() {
        let input = CameraInput::zeroed();
        assert_eq!(input, CameraInput::default());
        assert_close(input.move_forward, 0.0);
        assert_close(input.move_right, 0.0);
        assert_close(input.move_up, 0.0);
        assert_close(input.orbit_yaw, 0.0);
        assert_close(input.orbit_pitch, 0.0);
        assert_close(input.zoom, 0.0);
        assert_close(input.pan_right, 0.0);
        assert_close(input.pan_up, 0.0);
    }

    #[test]
    fn zeroed_input_leaves_the_camera_untouched() {
        let mut cam = EditorCamera::perspective();
        cam.orbit(0.4, -0.2);
        cam.move_forward(3.0);
        cam.zoom(0.5);
        let before = cam;
        CameraInput::zeroed().apply(&mut cam);
        assert_eq!(cam, before);
    }

    #[test]
    fn input_forward_move_travels_one_unit() {
        let mut cam = looking(0.6, 0.3);
        let start = cam.position;
        let forward = cam.forward();
        let mut input = CameraInput::zeroed();
        input.move_forward = 1.0;
        input.apply(&mut cam);
        assert_vec_close(cam.position, start + forward);
    }

    #[test]
    fn input_combines_move_orbit_and_zoom() {
        let mut cam = EditorCamera::perspective();
        let start = cam.position;
        let mut input = CameraInput::zeroed();
        input.move_right = 1.0;
        input.move_up = 2.0;
        input.orbit_yaw = 0.5;
        input.zoom = 1.0;
        input.apply(&mut cam);
        assert_close(cam.yaw, 0.5);
        assert_close(cam.pitch, 0.0);
        assert_vec_close(
            cam.position,
            start + Vec3::X + 2.0 * Vec3::Y + cam.forward(),
        );
    }

    #[test]
    fn input_pan_moves_along_the_axes_and_ignores_junk() {
        let mut cam = looking(1.2, 0.0);
        let start = cam.position;
        let mut input = CameraInput::zeroed();
        input.pan_right = 2.0;
        input.pan_up = -4.0;
        input.apply(&mut cam);
        assert_vec_close(cam.position, start + 2.0 * cam.right() - 4.0 * cam.up());
        let panned = cam;
        input.pan_right = f32::NAN;
        input.pan_up = f32::INFINITY;
        input.apply(&mut cam);
        assert_eq!(cam, panned);
    }

    #[test]
    fn input_zoom_clamps_like_the_camera_does() {
        let mut cam = EditorCamera::orthographic();
        let mut input = CameraInput::zeroed();
        input.zoom = -0.5;
        for _ in 0..1000 {
            input.apply(&mut cam);
        }
        assert_close(cam.orthographic_size, MIN_ORTHOGRAPHIC_SIZE);
    }

    #[test]
    fn camera_2d_starts_where_it_was_told() {
        let cam = Camera2DController::new(Vec2::new(3.0, -4.0), 64.0);
        assert_eq!(cam.center(), Vec2::new(3.0, -4.0));
        assert_close(cam.pixels_per_unit(), 64.0);
    }

    #[test]
    fn camera_2d_rejects_a_singular_scale() {
        for bad in [0.0, -1.0, -1.0e9, f32::NEG_INFINITY] {
            let cam = Camera2DController::new(Vec2::ZERO, bad);
            assert_close(cam.pixels_per_unit(), MIN_PIXELS_PER_UNIT);
            assert!(cam.pixels_per_unit() > 0.0);
        }
        let cam = Camera2DController::new(Vec2::ZERO, f32::NAN);
        assert_close(cam.pixels_per_unit(), DEFAULT_2D_PIXELS_PER_UNIT);
    }

    #[test]
    fn camera_2d_pan_moves_the_centre_by_exactly_the_delta() {
        let mut cam = Camera2DController::new(Vec2::new(1.0, 2.0), 32.0);
        cam.pan(3.0, -4.0);
        assert_eq!(cam.center(), Vec2::new(4.0, -2.0));
        cam.pan(-4.0, 2.0);
        assert_eq!(cam.center(), Vec2::new(0.0, 0.0));
        cam.pan(0.0, 0.0);
        assert_eq!(cam.center(), Vec2::ZERO);
        assert_close(cam.pixels_per_unit(), 32.0);
    }

    #[test]
    fn camera_2d_pan_accumulates_and_ignores_junk() {
        let mut cam = Camera2DController::new(Vec2::ZERO, 32.0);
        for _ in 0..10 {
            cam.pan(0.5, -0.25);
        }
        assert_eq!(cam.center(), Vec2::new(5.0, -2.5));
        let panned = cam.center();
        cam.pan(f32::NAN, f32::INFINITY);
        assert_eq!(cam.center(), panned);
    }

    #[test]
    fn camera_2d_zoom_in_is_monotonic() {
        let mut cam = Camera2DController::new(Vec2::ZERO, 1.0);
        let mut previous = cam.pixels_per_unit();
        for _ in 0..5 {
            let factor = cam.zoom_at(0.5);
            assert!(factor > 1.0, "factor {factor}");
            assert!(
                cam.pixels_per_unit() > previous,
                "{} !> {}",
                cam.pixels_per_unit(),
                previous
            );
            previous = cam.pixels_per_unit();
        }
    }

    #[test]
    fn camera_2d_zoom_out_is_monotonic_until_the_clamp() {
        let mut cam = Camera2DController::new(Vec2::ZERO, 1024.0);
        let mut previous = cam.pixels_per_unit();
        while cam.pixels_per_unit() > MIN_PIXELS_PER_UNIT {
            let factor = cam.zoom_at(-0.5);
            assert!(factor < 1.0, "factor {factor}");
            assert!(cam.pixels_per_unit() < previous);
            previous = cam.pixels_per_unit();
        }
        assert_close(cam.pixels_per_unit(), MIN_PIXELS_PER_UNIT);
        for _ in 0..1000 {
            assert_eq!(cam.zoom_at(-0.5), 1.0, "the clamp must pin the factor");
            assert_close(cam.pixels_per_unit(), MIN_PIXELS_PER_UNIT);
        }
        assert!(cam.pixels_per_unit() > 0.0);
    }

    #[test]
    fn camera_2d_zoom_clamps_a_singular_request() {
        let mut cam = Camera2DController::new(Vec2::ZERO, 8.0);
        for _ in 0..1000 {
            cam.zoom_at(-1.0);
        }
        assert_close(cam.pixels_per_unit(), MIN_PIXELS_PER_UNIT);
        assert!(cam.pixels_per_unit() > 0.0);
        assert_finite(cam.view_projection(Vec2::new(640.0, 480.0)));
    }

    #[test]
    fn camera_2d_zoom_ignores_junk_and_never_goes_non_finite() {
        let mut cam = Camera2DController::new(Vec2::ZERO, 32.0);
        let before = cam.pixels_per_unit();
        assert_close(cam.zoom_at(f32::NAN), 1.0);
        assert_close(cam.pixels_per_unit(), before);
        assert_close(cam.zoom_at(0.0), 1.0);
        assert_close(cam.pixels_per_unit(), before);
        cam.zoom_at(f32::INFINITY);
        assert_close(cam.pixels_per_unit(), before);
        cam.zoom_at(f32::MAX / before);
        assert_close(cam.pixels_per_unit(), f32::MAX);
        assert_finite(cam.view_projection(Vec2::new(800.0, 600.0)));
    }

    #[test]
    fn camera_2d_zoom_factor_matches_the_applied_scale() {
        let mut cam = Camera2DController::new(Vec2::ZERO, 10.0);
        let before = cam.pixels_per_unit();
        let factor = cam.zoom_at(1.5);
        assert_close(factor, 2.5);
        assert_close(cam.pixels_per_unit(), before * 2.5);
    }

    #[test]
    fn camera_2d_view_projection_is_finite() {
        let cam = Camera2DController::new(Vec2::new(7.0, -3.0), 48.0);
        for viewport in [
            Vec2::new(1920.0, 1080.0),
            Vec2::new(1.0, 1.0),
            Vec2::new(0.0, 0.0),
            Vec2::new(-10.0, 1080.0),
            Vec2::new(f32::NAN, f32::INFINITY),
        ] {
            assert_finite(cam.view_projection(viewport));
        }
    }

    #[test]
    fn camera_2d_view_projection_centre_is_the_clip_origin() {
        let cam = Camera2DController::new(Vec2::new(12.5, -6.25), 32.0);
        let clip =
            cam.view_projection(Vec2::new(800.0, 600.0)) * cam.center().extend(0.0).extend(1.0);
        assert!(clip.x.abs() < 1e-4, "{clip:?}");
        assert!(clip.y.abs() < 1e-4, "{clip:?}");
        assert_close(clip.w, 1.0);
    }

    #[test]
    fn camera_2d_view_projection_is_y_up() {
        let cam = Camera2DController::new(Vec2::ZERO, 100.0);
        let matrix = cam.view_projection(Vec2::new(800.0, 600.0));
        let top_right = ndc2(matrix, Vec2::new(4.0, 3.0));
        assert!(top_right.y > 0.0, "{top_right:?}");
        assert!(top_right.x > 0.0, "{top_right:?}");
    }

    #[test]
    fn camera_2d_view_projection_fills_the_viewport() {
        let cam = Camera2DController::new(Vec2::ZERO, 50.0);
        let viewport = Vec2::new(800.0, 600.0);
        let matrix = cam.view_projection(viewport);
        let half = viewport * (0.5 / cam.pixels_per_unit());
        let corner = ndc2(matrix, half);
        assert_close(corner.x, 1.0);
        assert_close(corner.y, 1.0);
    }

    #[test]
    fn camera_2d_view_projection_keeps_the_centre_fixed_while_zooming() {
        let mut cam = Camera2DController::new(Vec2::new(2.0, 5.0), 16.0);
        let factor = cam.zoom_at(3.0);
        assert_close(factor, 4.0);
        let clip =
            cam.view_projection(Vec2::new(320.0, 240.0)) * cam.center().extend(0.0).extend(1.0);
        assert!(clip.x.abs() < 1e-4 && clip.y.abs() < 1e-4, "{clip:?}");
    }

    #[test]
    fn camera_2d_view_projection_scale_doubles_with_the_zoom() {
        let viewport = Vec2::new(800.0, 600.0);
        let small = Camera2DController::new(Vec2::ZERO, 25.0).view_projection(viewport);
        let large = Camera2DController::new(Vec2::ZERO, 50.0).view_projection(viewport);
        assert_close(large.x_axis.x, small.x_axis.x * 2.0);
        assert_close(large.y_axis.y, small.y_axis.y * 2.0);
    }

    #[test]
    fn camera_2d_view_projection_fills_the_halved_viewport() {
        let cam = Camera2DController::new(Vec2::ZERO, 40.0);
        let point = Vec2::new(5.0, 3.75);
        let wide = ndc2(cam.view_projection(Vec2::new(800.0, 600.0)), point);
        let narrow = ndc2(cam.view_projection(Vec2::new(400.0, 300.0)), point);
        assert_close(wide.x, 0.5);
        assert_close(wide.y, 0.5);
        assert_close(narrow.x, 1.0);
        assert_close(narrow.y, 1.0);
    }

    #[test]
    fn camera_2d_pan_shifts_the_projection() {
        let mut cam = Camera2DController::new(Vec2::ZERO, 50.0);
        let viewport = Vec2::new(500.0, 500.0);
        let point = Vec3::new(1.0, 0.0, 0.0);
        let before = ndc(cam.view_projection(viewport), point);
        cam.pan(1.0, 0.0);
        let after = ndc(cam.view_projection(viewport), point);
        assert!(before.x > 0.0 && after.x < before.x, "{before:?} {after:?}");
    }

    #[test]
    fn camera_2d_view_projection_maps_the_depth_range() {
        let cam = Camera2DController::new(Vec2::ZERO, 32.0);
        let matrix = cam.view_projection(Vec2::new(100.0, 100.0));
        let flat = matrix * Vec3::new(0.0, 0.0, 0.0).extend(1.0);
        let back = matrix * Vec3::new(0.0, 0.0, -1.0).extend(1.0);
        assert_close(flat.z, 0.0);
        assert_close(back.z, 1.0);
        assert_close(flat.w, 1.0);
        assert_close(back.w, 1.0);
    }
}
