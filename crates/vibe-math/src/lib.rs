//! Math types for vibeEngine.
//!
//! Re-exports the [`glam`] vector/quaternion/matrix types so downstream crates
//! have a single import path, and adds the few engine-specific helpers that
//! have no upstream equivalent.

#[cfg(test)]
use glam::Mat3;
use glam::camera::rh::proj::vulkan::perspective_infinite_reverse;
use glam::camera::rh::view::{look_at_mat4, look_to_mat4};
use glam::{Mat4, Quat, Vec2, Vec3, Vec4};

/// A 2D axis-aligned bounding box.
///
/// The origin is the top-left corner in engine space (Y down, matching screen
/// coordinates); callers convert to a centre form with [`Aabb2::center`] and
/// [`Aabb2::half_extents`] when talking to physics engines that centre boxes.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Aabb2 {
    /// Smallest corner.
    pub min: Vec2,
    /// Largest corner.
    pub max: Vec2,
}

impl Aabb2 {
    /// A box spanning `min` to `max`.
    pub fn new(min: Vec2, max: Vec2) -> Self {
        Self { min, max }
    }

    /// A box centred on the origin with the given full width and height.
    pub fn from_center_half_extents(center: Vec2, half_extents: Vec2) -> Self {
        Self {
            min: center - half_extents,
            max: center + half_extents,
        }
    }

    /// The centre point.
    pub fn center(&self) -> Vec2 {
        (self.min + self.max) * 0.5
    }

    /// Half the width and height.
    pub fn half_extents(&self) -> Vec2 {
        (self.max - self.min) * 0.5
    }

    /// Full width and height.
    pub fn size(&self) -> Vec2 {
        self.max - self.min
    }

    /// True when `point` is inside the box, edges included.
    pub fn contains(&self, point: Vec2) -> bool {
        point.cmpge(self.min).all() && point.cmple(self.max).all()
    }

    /// The smallest box containing both `self` and `other`.
    pub fn union(&self, other: &Aabb2) -> Aabb2 {
        Aabb2 {
            min: self.min.min(other.min),
            max: self.max.max(other.max),
        }
    }
}

/// A 3D axis-aligned bounding box.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Aabb3 {
    /// Smallest corner.
    pub min: Vec3,
    /// Largest corner.
    pub max: Vec3,
}

impl Aabb3 {
    /// A box spanning `min` to `max`.
    pub fn new(min: Vec3, max: Vec3) -> Self {
        Self { min, max }
    }

    /// A box centred on the origin with the given half extents.
    pub fn from_center_half_extents(center: Vec3, half_extents: Vec3) -> Self {
        Self {
            min: center - half_extents,
            max: center + half_extents,
        }
    }

    /// The centre point.
    pub fn center(&self) -> Vec3 {
        (self.min + self.max) * 0.5
    }

    /// Half the width, height and depth.
    pub fn half_extents(&self) -> Vec3 {
        (self.max - self.min) * 0.5
    }

    /// Full width, height and depth.
    pub fn size(&self) -> Vec3 {
        self.max - self.min
    }

    /// True when `point` is inside the box, edges included.
    pub fn contains(&self, point: Vec3) -> bool {
        point.cmpge(self.min).all() && point.cmple(self.max).all()
    }

    /// The smallest box containing both `self` and `other`.
    pub fn union(&self, other: &Aabb3) -> Aabb3 {
        Aabb3 {
            min: self.min.min(other.min),
            max: self.max.max(other.max),
        }
    }
}

/// A right-handed look-at view matrix, matching `glam::Mat4::look_at_rh`.
///
/// Provided as a free function so call sites read the same whether or not they
/// have imported `glam` itself.
pub fn look_at_rh(eye: Vec3, target: Vec3, up: Vec3) -> Mat4 {
    look_at_mat4(eye, target, up)
}

/// A right-handed reverse-look-at camera transform.
///
/// Yields a world transform to hand to `Mat4::inverse` for a view matrix.
pub fn look_to_rh(eye: Vec3, target: Vec3, up: Vec3) -> Mat4 {
    look_to_mat4(eye, (target - eye).normalize(), up)
}

/// A reverse-Z infinite-far perspective projection.
///
/// The near plane maps to `1.0` and infinity to `0.0`, which maximises float
/// precision in the depth buffer for large view distances.
pub fn perspective_rh_reverse_z_infinite(fov_y_radians: f32, aspect: f32, near: f32) -> Mat4 {
    perspective_infinite_reverse(fov_y_radians, aspect, near)
}

/// An orthographic projection with a right-handed, zero-to-one depth range.
///
/// Built by hand rather than via glam's `vulkan::orthographic`, whose Y-flip
/// flag inverts the vertical axis and produces negative depth for the
/// zero-to-near range a 2D camera needs. Depth expects the view space's
/// forward direction, so a point in front sits at negative Z.
pub fn orthographic_rh(left: f32, right: f32, bottom: f32, top: f32, near: f32, far: f32) -> Mat4 {
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

/// Interpolate between two transforms, rotating along the shortest arc.
///
/// `t` is clamped to `0.0..=1.0`, so callers cannot produce a non-finite
/// transform by overshooting the blend factor.
pub fn lerp_transform(a: Mat4, b: Mat4, t: f32) -> Mat4 {
    let t = t.clamp(0.0, 1.0);
    // to_scale_rotation_translation returns (scale, rotation, translation).
    let (a_scale, a_rot, a_pos) = a.to_scale_rotation_translation();
    let (b_scale, b_rot, b_pos) = b.to_scale_rotation_translation();
    let rot = Quat::slerp(a_rot, b_rot, t);
    Mat4::from_scale_rotation_translation(a_scale.lerp(b_scale, t), rot, a_pos.lerp(b_pos, t))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aabb2_center_and_extents() {
        let b = Aabb2::new(Vec2::new(-2.0, -4.0), Vec2::new(6.0, 4.0));
        assert_eq!(b.center(), Vec2::new(2.0, 0.0));
        assert_eq!(b.half_extents(), Vec2::new(4.0, 4.0));
        assert_eq!(b.size(), Vec2::new(8.0, 8.0));
    }

    #[test]
    fn aabb2_contains_includes_edges() {
        let b = Aabb2::new(Vec2::ZERO, Vec2::ONE);
        assert!(b.contains(Vec2::new(0.5, 0.5)));
        assert!(b.contains(Vec2::ZERO));
        assert!(b.contains(Vec2::ONE));
        assert!(!b.contains(Vec2::new(1.001, 0.5)));
    }

    #[test]
    fn aabb2_from_center_matches_new() {
        let a = Aabb2::new(Vec2::new(-1.0, -3.0), Vec2::new(5.0, 3.0));
        let b = Aabb2::from_center_half_extents(a.center(), a.half_extents());
        assert_eq!(a, b);
    }

    #[test]
    fn aabb2_union_covers_both() {
        let a = Aabb2::new(Vec2::ZERO, Vec2::splat(2.0));
        let b = Aabb2::new(Vec2::splat(-5.0), Vec2::splat(1.0));
        let u = a.union(&b);
        assert_eq!(u.min, Vec2::splat(-5.0));
        assert_eq!(u.max, Vec2::splat(2.0));
    }

    #[test]
    fn aabb3_geometry() {
        let b = Aabb3::from_center_half_extents(Vec3::new(1.0, 2.0, 3.0), Vec3::ONE);
        assert_eq!(b.size(), Vec3::splat(2.0));
        assert_eq!(b.center(), Vec3::new(1.0, 2.0, 3.0));
        assert!(b.contains(Vec3::new(0.0, 2.0, 4.0)));
        assert!(!b.contains(Vec3::new(5.0, 2.0, 3.0)));
    }

    #[test]
    fn aabb3_union_covers_both() {
        let a = Aabb3::new(Vec3::ZERO, Vec3::splat(1.0));
        let b = Aabb3::new(Vec3::splat(3.0), Vec3::splat(4.0));
        let u = a.union(&b);
        assert_eq!(u.min, Vec3::ZERO);
        assert_eq!(u.max, Vec3::splat(4.0));
    }

    #[test]
    fn reverse_z_infinite_near_is_one() {
        // Right-handed view space puts points in front of the camera at
        // negative Z; reverse-Z then maps near to 1.0 and infinity to 0.0.
        let proj = perspective_rh_reverse_z_infinite(1.0, 1.0, 0.1);
        let near = proj * Vec4::new(0.0, 0.0, -0.1, 1.0);
        let ndc_near = near.truncate() / near.w;
        assert!(
            (ndc_near.z - 1.0).abs() < 1e-5,
            "near plane should map to 1.0, got {}",
            ndc_near.z
        );

        let far = proj * Vec4::new(0.0, 0.0, -1.0e6, 1.0);
        let ndc_far = far.truncate() / far.w;
        assert!(
            ndc_far.z.abs() < 1e-5,
            "far plane should approach 0.0, got {}",
            ndc_far.z
        );
    }

    #[test]
    fn orthographic_maps_near_and_far_to_unit_range() {
        // Near and far are distances along -Z in this view space.
        let proj = orthographic_rh(-1.0, 1.0, -1.0, 1.0, 0.5, 10.0);
        let n = proj * Vec4::new(0.0, 0.0, -0.5, 1.0);
        assert!((n.z / n.w).abs() < 1e-6, "near should map to 0");
        let f = proj * Vec4::new(0.0, 0.0, -10.0, 1.0);
        assert!(((f.z / f.w) - 1.0).abs() < 1e-6, "far should map to 1");
    }

    #[test]
    fn lerp_transform_endpoints_are_close() {
        let a = Mat4::from_scale_rotation_translation(Vec3::ONE, Quat::IDENTITY, Vec3::ZERO);
        let b = Mat4::from_scale_rotation_translation(
            Vec3::splat(3.0),
            Quat::from_rotation_y(1.0),
            Vec3::new(4.0, 5.0, 6.0),
        );
        // Decomposing and recomposing loses the last bits of the quaternion,
        // so the endpoints are compared within a tolerance rather than exactly.
        for t in [0.0f32, 1.0] {
            let got = lerp_transform(a, b, t);
            let want = if t == 0.0 { a } else { b };
            assert!(
                (got.x_axis.x - want.x_axis.x).abs() < 1e-5
                    && (got.w_axis.x - want.w_axis.x).abs() < 1e-5,
                "t={t}: {got:?} != {want:?}"
            );
        }
    }

    #[test]
    fn lerp_transform_midpoint() {
        let a = Mat4::from_scale_rotation_translation(Vec3::ONE, Quat::IDENTITY, Vec3::ZERO);
        let b = Mat4::from_scale_rotation_translation(Vec3::ONE, Quat::IDENTITY, Vec3::splat(10.0));
        let mid = lerp_transform(a, b, 0.5);
        assert!((mid.w_axis.truncate().x - 5.0).abs() < 1e-5);
    }

    #[test]
    fn lerp_transform_clamps_out_of_range_t() {
        let a = Mat4::IDENTITY;
        let b = Mat4::from_scale_rotation_translation(Vec3::splat(2.0), Quat::IDENTITY, Vec3::ZERO);
        assert_eq!(lerp_transform(a, b, -1.0), a);
        assert_eq!(lerp_transform(a, b, 2.0), b);
    }

    #[test]
    fn look_at_places_eye_at_origin_of_view() {
        let view = look_at_rh(Vec3::new(0.0, 0.0, 5.0), Vec3::ZERO, Vec3::Y);
        let eye_in_view = view * Vec4::new(0.0, 0.0, 5.0, 1.0);
        assert!((eye_in_view.x).abs() < 1e-5);
        assert!((eye_in_view.y).abs() < 1e-5);
    }

    #[test]
    fn look_to_matches_look_at_for_the_same_direction() {
        // look_to takes a direction, so it is a look_at with an already
        // normalised vector; both must agree.
        let eye = Vec3::new(3.0, 4.0, 5.0);
        let target = Vec3::new(-1.0, 0.0, 2.0);
        let up = Vec3::Y;
        let a = look_at_rh(eye, target, up);
        let b = look_to_rh(eye, target, up);
        assert!((a.x_axis.x - b.x_axis.x).abs() < 1e-4);
        assert!((a.w_axis.z - b.w_axis.z).abs() < 1e-4);
    }

    #[test]
    fn mat3_is_exported_for_normal_matrices() {
        let m: Mat3 = Mat3::from_mat4(Mat4::IDENTITY);
        assert_eq!(m, Mat3::IDENTITY);
    }
}
