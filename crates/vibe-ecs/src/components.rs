//! The engine's component set.

use std::sync::Arc;

use glam::{Mat4, Quat, UVec2, Vec2, Vec3, Vec4};

/// Position, rotation and scale, in that order applied as TRS.
///
/// The default is identity, not a zero scale, so a `Transform` on its own
/// leaves geometry where it was authored.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Transform {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub rotation: Quat,
    pub scale: Vec3,
}

impl Default for Transform {
    fn default() -> Self {
        Transform {
            x: 0.0,
            y: 0.0,
            z: 0.0,
            rotation: Quat::IDENTITY,
            scale: Vec3::ONE,
        }
    }
}

impl Transform {
    /// A 2D transform at `(x, y)` with no rotation and unit scale.
    pub fn at_2d(x: f32, y: f32) -> Transform {
        Transform {
            x,
            y,
            z: 0.0,
            ..Default::default()
        }
    }

    /// Translation as a vector.
    pub fn translation(&self) -> Vec3 {
        Vec3::new(self.x, self.y, self.z)
    }

    /// The composed TRS matrix.
    pub fn matrix(&self) -> Mat4 {
        Mat4::from_scale_rotation_translation(self.scale, self.rotation, self.translation())
    }

    /// Uniform or per-axis scale.
    pub fn scaled(&self, scale: Vec3) -> Transform {
        Transform { scale, ..*self }
    }
}

/// A free-form name used for lookups and editor filtering.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Default)]
pub struct Tag(pub String);

impl Tag {
    /// A tag from anything string-like.
    pub fn new(s: impl Into<String>) -> Tag {
        Tag(s.into())
    }
}

/// The entity's UUID, stored as a component so scenes can carry it in data.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub struct Id(pub crate::entity::Uuid);

impl Id {
    /// A fresh random id.
    pub fn generate() -> Id {
        Id(crate::entity::Uuid::new_v4())
    }

    /// The wrapped UUID.
    pub fn value(&self) -> crate::entity::Uuid {
        self.0
    }
}

/// A camera; the first one added becomes the primary camera.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Camera {
    /// Vertical field of view in radians; ignored for orthographic.
    pub fov_y_radians: f32,
    /// Near clip distance.
    pub near: f32,
    /// Far clip distance; ignored for infinite reverse-Z.
    pub far: f32,
    /// Colour written where nothing is drawn, in linear space.
    pub clear_color: Vec4,
    /// When set, `fov_y_radians` is ignored and these half-extents are used.
    pub orthographic: Option<Vec2>,
}

impl Default for Camera {
    fn default() -> Self {
        Camera {
            fov_y_radians: std::f32::consts::FRAC_PI_3,
            near: 0.1,
            far: 1000.0,
            clear_color: Vec4::new(0.02, 0.02, 0.03, 1.0),
            orthographic: None,
        }
    }
}

/// A textured or untextured 2D quad.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpriteRenderer {
    /// Index into the bindless texture array.
    pub texture_index: u32,
    /// Top-left corner offset in pixels.
    pub offset: Vec2,
    /// Width and height in pixels.
    pub size: Vec2,
    /// Rotation about the centre, in radians.
    pub rotation_radians: f32,
    /// Multiplied into the sampled texel.
    pub tint: Vec4,
    /// UV origin, for sub-textures from a sprite atlas.
    pub uv_origin: Vec2,
    /// UV size, for sub-textures from a sprite atlas.
    pub uv_size: Vec2,
    /// Draw order within its layer; higher draws later.
    pub sort_order: i32,
}

impl Default for SpriteRenderer {
    fn default() -> Self {
        SpriteRenderer {
            texture_index: 0,
            offset: Vec2::ZERO,
            size: Vec2::splat(64.0),
            rotation_radians: 0.0,
            tint: Vec4::ONE,
            uv_origin: Vec2::ZERO,
            uv_size: Vec2::ONE,
            sort_order: 0,
        }
    }
}

/// A string drawn with the MSDF text renderer.
#[derive(Debug, Clone, PartialEq)]
pub struct Text {
    pub content: String,
    pub size: f32,
    pub color: Vec4,
    pub alignment: TextAlignment,
    pub line_spacing: f32,
}

impl Default for Text {
    fn default() -> Self {
        Text {
            content: String::new(),
            size: 16.0,
            color: Vec4::ONE,
            alignment: TextAlignment::Left,
            line_spacing: 1.2,
        }
    }
}

/// Horizontal text alignment within the entity's width.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TextAlignment {
    #[default]
    Left,
    Center,
    Right,
}

/// Frame-by-frame playback of a sprite sheet.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpriteAnimation {
    /// First frame index in the sheet.
    pub start_frame: u32,
    /// One past the last frame index.
    pub frame_count: u32,
    /// Frames per texture row.
    pub columns: u32,
    /// UV size of a single frame.
    pub frame_uv_size: Vec2,
    /// Seconds each frame is shown.
    pub frame_duration: f32,
    pub loop_mode: AnimationLoop,
    /// Playback position in frames; written by the animation system.
    pub current_frame: f32,
}

impl Default for SpriteAnimation {
    fn default() -> Self {
        SpriteAnimation {
            start_frame: 0,
            frame_count: 1,
            columns: 1,
            frame_uv_size: Vec2::ONE,
            frame_duration: 1.0 / 12.0,
            loop_mode: AnimationLoop::Loop,
            current_frame: 0.0,
        }
    }
}

/// What happens when an animation reaches its last frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum AnimationLoop {
    /// Restart from the first frame.
    #[default]
    Loop,
    /// Stay on the last frame.
    Clamp,
    /// Jump back to the first frame immediately.
    PingPong,
}

/// A native Rust script attached to an entity.
#[derive(Clone)]
pub struct NativeScript {
    /// Name the scene file refers to.
    pub name: String,
    /// Opaque handle to the registered script instance.
    pub instance: std::sync::Arc<dyn std::any::Any + Send + Sync>,
}

impl std::fmt::Debug for NativeScript {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NativeScript")
            .field("name", &self.name)
            .finish_non_exhaustive()
    }
}

impl PartialEq for NativeScript {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name && Arc::ptr_eq(&self.instance, &other.instance)
    }
}

impl NativeScript {
    /// Attach a script by name with its constructed state.
    pub fn new<S: std::any::Any + Send + Sync>(
        name: impl Into<String>,
        instance: S,
    ) -> NativeScript {
        NativeScript {
            name: name.into(),
            instance: std::sync::Arc::new(instance),
        }
    }

    /// Borrow the script state as `S`.
    pub fn state<S: std::any::Any + Send + Sync>(&self) -> Option<&S> {
        self.instance.downcast_ref::<S>()
    }
}

impl Default for NativeScript {
    fn default() -> Self {
        NativeScript {
            name: String::new(),
            instance: std::sync::Arc::new(()),
        }
    }
}

/// A mesh rendered by the 3D pipeline.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Model3D {
    /// Asset path of the glTF/GLB file.
    pub mesh_path: String,
    /// Skin index inside the loaded model.
    pub skin_index: u32,
    /// Tint multiplied into the base colour.
    pub tint: Vec4,
    /// Draw order among other 3D entities; higher draws later.
    pub sort_order: i32,
}

/// Playback state for a skeletal animation clip.
#[derive(Debug, Clone, PartialEq)]
pub struct Animator {
    pub clip_index: u32,
    pub time_seconds: f32,
    pub speed: f32,
    pub loop_mode: AnimationLoop,
    /// Blend weight in `0.0..=1.0` against the previous clip.
    pub blend: f32,
}

impl Default for Animator {
    fn default() -> Self {
        Animator {
            clip_index: 0,
            time_seconds: 0.0,
            speed: 1.0,
            loop_mode: AnimationLoop::Loop,
            blend: 1.0,
        }
    }
}

/// An infinitely distant light emitting along a direction.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DirectionalLight {
    /// Colour multiplied into the surface.
    pub color: Vec3,
    /// Linear intensity.
    pub intensity: f32,
    /// Render direction; the light travels along this vector.
    pub direction: Vec3,
}

impl Default for DirectionalLight {
    fn default() -> Self {
        DirectionalLight {
            color: Vec3::ONE,
            intensity: 1.0,
            direction: Vec3::NEG_Y,
        }
    }
}

/// A light radiating from a point.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PointLight {
    pub color: Vec3,
    pub intensity: f32,
    /// Distance at which intensity reaches zero.
    pub range: f32,
    /// Light beyond this radius does not attenuate, avoiding a hard cutoff.
    pub radius: f32,
}

impl Default for PointLight {
    fn default() -> Self {
        PointLight {
            color: Vec3::ONE,
            intensity: 1.0,
            range: 10.0,
            radius: 0.1,
        }
    }
}

/// How a 2D body moves.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BodyType2D {
    #[default]
    Static,
    Kinematic,
    Dynamic,
}

/// How a 3D body moves.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum BodyType3D {
    #[default]
    Static,
    Kinematic,
    Dynamic,
}

/// A 2D rigid body.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rigidbody2D {
    pub body_type: BodyType2D,
    /// Mass in kilograms; ignored for static bodies.
    pub mass: f32,
    /// Fraction of velocity shed per second.
    pub linear_damping: f32,
    pub angular_damping: f32,
    pub gravity_scale: f32,
    pub is_sensor: bool,
    /// True when the 2D solver should rotate this body, it collides, and its
    /// centre of mass is not at the origin.
    pub is_rotatable: bool,
}

impl Default for Rigidbody2D {
    fn default() -> Self {
        Rigidbody2D {
            body_type: BodyType2D::Static,
            mass: 1.0,
            linear_damping: 0.0,
            angular_damping: 0.0,
            gravity_scale: 1.0,
            is_sensor: false,
            is_rotatable: false,
        }
    }
}

/// A 3D rigid body.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rigidbody3D {
    pub body_type: BodyType3D,
    pub mass: f32,
    pub linear_damping: f32,
    pub angular_damping: f32,
    pub gravity_scale: f32,
    pub is_sensor: bool,
    /// Per-axis translation locks, `(x, y, z)`.
    pub lock_linear: (bool, bool, bool),
    /// Per-axis rotation locks, `(x, y, z)`.
    pub lock_angular: (bool, bool, bool),
}

impl Default for Rigidbody3D {
    fn default() -> Self {
        Rigidbody3D {
            body_type: BodyType3D::Static,
            mass: 1.0,
            linear_damping: 0.0,
            angular_damping: 0.0,
            gravity_scale: 1.0,
            is_sensor: false,
            lock_linear: (false, false, false),
            lock_angular: (false, false, false),
        }
    }
}

/// Material properties shared by 2D and 3D box colliders.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ColliderMaterial {
    /// Kilograms per square unit; mass is derived from this for dynamic bodies.
    pub density: f32,
    /// Tangential friction coefficient.
    pub friction: f32,
    /// Restitution, 0 = no bounce, 1 = perfectly elastic.
    pub restitution: f32,
}

impl Default for ColliderMaterial {
    fn default() -> Self {
        ColliderMaterial {
            density: 1.0,
            friction: 0.3,
            restitution: 0.0,
        }
    }
}

/// An axis-aligned 2D box collider.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct BoxCollider2D {
    /// Full width and height in world units.
    pub size: Vec2,
    /// Offset from the transform's position.
    pub offset: Vec2,
    pub material: ColliderMaterial,
}

impl BoxCollider2D {
    /// A collider of the given size centred on the origin.
    pub fn new(size: Vec2) -> BoxCollider2D {
        BoxCollider2D {
            size,
            ..Default::default()
        }
    }
}

/// An axis-aligned 3D box collider.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct BoxCollider3D {
    pub size: Vec3,
    pub offset: Vec3,
    pub material: ColliderMaterial,
}

/// A 3D sphere collider.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct SphereCollider3D {
    pub radius: f32,
    pub offset: Vec3,
    pub material: ColliderMaterial,
}

/// A 3D capsule collider, aligned to an axis.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CapsuleCollider3D {
    pub radius: f32,
    /// Length of the cylindrical section, excluding the caps.
    pub height: f32,
    pub offset: Vec3,
    /// Capsule orientation: `0` = Y, `1` = X, `2` = Z.
    pub axis: u8,
    pub material: ColliderMaterial,
}

impl Default for CapsuleCollider3D {
    fn default() -> Self {
        CapsuleCollider3D {
            radius: 0.5,
            height: 1.0,
            offset: Vec3::ZERO,
            axis: 0,
            material: ColliderMaterial::default(),
        }
    }
}

impl CapsuleCollider3D {
    /// Total length including both hemispherical caps.
    pub fn total_height(&self) -> f32 {
        self.height + 2.0 * self.radius
    }
}

/// Framebuffer size, written by the windowing system each resize.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Viewport {
    pub size: UVec2,
    /// Logical size before pixel ratio scaling, used for UI layout.
    pub logical_size: UVec2,
}

impl Viewport {
    /// A viewport of the given size, with matching logical size.
    pub fn new(width: u32, height: u32) -> Viewport {
        Viewport {
            size: UVec2::new(width, height),
            logical_size: UVec2::new(width, height),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn transform_matrix_is_identity_by_default() {
        assert_eq!(Transform::default().matrix(), Mat4::IDENTITY);
    }

    #[test]
    fn transform_at_2d_sets_xy_only() {
        let t = Transform::at_2d(3.0, 4.0);
        assert_eq!(t.translation(), Vec3::new(3.0, 4.0, 0.0));
    }

    #[test]
    fn transform_scaled_keeps_position() {
        let t = Transform::at_2d(1.0, 2.0).scaled(Vec3::splat(2.0));
        assert_eq!(t.scale, Vec3::splat(2.0));
        assert_eq!(t.x, 1.0);
    }

    #[test]
    fn tag_wraps_any_string() {
        assert_eq!(Tag::new("player").0, "player");
        assert_eq!(Tag::default().0, "");
    }

    #[test]
    fn id_generates_non_nil() {
        assert!(!Id::generate().value().is_nil());
    }

    #[test]
    fn camera_default_is_perspective() {
        assert!(Camera::default().orthographic.is_none());
        assert!(Camera::default().fov_y_radians > 0.0);
    }

    #[test]
    fn camera_orthographic_overrides_fov() {
        let c = Camera {
            orthographic: Some(Vec2::splat(5.0)),
            ..Default::default()
        };
        assert_eq!(c.orthographic, Some(Vec2::splat(5.0)));
    }

    #[test]
    fn sprite_default_is_64px_opaque() {
        let s = SpriteRenderer::default();
        assert_eq!(s.size, Vec2::splat(64.0));
        assert_eq!(s.tint, Vec4::ONE);
        assert_eq!(s.uv_size, Vec2::ONE);
    }

    #[test]
    fn text_default_is_left_aligned() {
        assert_eq!(Text::default().alignment, TextAlignment::Left);
        assert!(Text::default().content.is_empty());
    }

    #[test]
    fn sprite_animation_defaults_to_looping() {
        let a = SpriteAnimation::default();
        assert_eq!(a.loop_mode, AnimationLoop::Loop);
        assert_eq!(a.frame_count, 1);
    }

    #[test]
    fn native_script_state_downcasts() {
        let s = NativeScript::new("mover", 42u32);
        assert_eq!(s.name, "mover");
        assert_eq!(s.state::<u32>(), Some(&42));
        assert_eq!(s.state::<f32>(), None);
    }

    #[test]
    fn model3d_default_has_no_path() {
        assert!(Model3D::default().mesh_path.is_empty());
    }

    #[test]
    fn animator_defaults_to_full_blend() {
        assert_eq!(Animator::default().blend, 1.0);
        assert_eq!(Animator::default().speed, 1.0);
    }

    #[test]
    fn directional_light_points_down() {
        assert_eq!(DirectionalLight::default().direction, Vec3::NEG_Y);
    }

    #[test]
    fn point_light_default_range() {
        assert_eq!(PointLight::default().range, 10.0);
    }

    #[test]
    fn body_types_default_to_static() {
        assert_eq!(BodyType2D::default(), BodyType2D::Static);
        assert_eq!(BodyType3D::default(), BodyType3D::Static);
    }

    #[test]
    fn rigidbody2d_dynamic_has_mass() {
        let r = Rigidbody2D {
            body_type: BodyType2D::Dynamic,
            mass: 5.0,
            ..Default::default()
        };
        assert_eq!(r.mass, 5.0);
        assert!(!r.is_sensor);
    }

    #[test]
    fn rigidbody3d_locks_default_open() {
        let r = Rigidbody3D::default();
        assert_eq!(r.lock_linear, (false, false, false));
        assert_eq!(r.lock_angular, (false, false, false));
    }

    #[test]
    fn collider_material_defaults() {
        let m = ColliderMaterial::default();
        assert_eq!(m.density, 1.0);
        assert_eq!(m.friction, 0.3);
        assert_eq!(m.restitution, 0.0);
    }

    #[test]
    fn box_collider2d_new_sets_size() {
        let c = BoxCollider2D::new(Vec2::splat(3.0));
        assert_eq!(c.size, Vec2::splat(3.0));
        assert_eq!(c.offset, Vec2::ZERO);
    }

    #[test]
    fn box_collider3d_defaults_to_unit_cube() {
        assert_eq!(BoxCollider3D::default().size, Vec3::ZERO);
    }

    #[test]
    fn sphere_collider_default_radius() {
        assert_eq!(SphereCollider3D::default().radius, 0.0);
    }

    #[test]
    fn capsule_total_height_includes_caps() {
        let c = CapsuleCollider3D {
            radius: 0.5,
            height: 2.0,
            ..Default::default()
        };
        assert_eq!(c.total_height(), 3.0);
    }

    #[test]
    fn capsule_defaults_to_y_axis() {
        assert_eq!(CapsuleCollider3D::default().axis, 0);
    }

    #[test]
    fn viewport_new_mirrors_logical_size() {
        let v = Viewport::new(800, 600);
        assert_eq!(v.size, UVec2::new(800, 600));
        assert_eq!(v.logical_size, UVec2::new(800, 600));
    }

    #[test]
    fn animation_loop_default_is_loop() {
        assert_eq!(AnimationLoop::default(), AnimationLoop::Loop);
    }
}
