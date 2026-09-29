//! YAML scene serialization, both directions.
//!
//! The on-disk form is a flat list of entities, each with a UUID and a map of
//! component names to values. Keeping it flat and explicit means a hand-edited
//! scene file stays readable, which matters more than compactness for content
//! people actually open.

use glam::{Quat, Vec2, Vec3, Vec4};
use serde::{Deserialize, Serialize};
// Imported one by one: the ComponentValue variants below are named after the
// same types, so a glob would be shadowed by them.
use vibe_ecs::components::{
    AnimationLoop, Animator, BodyType2D, BodyType3D, BoxCollider2D, BoxCollider3D, Camera,
    CapsuleCollider3D, ColliderMaterial, DirectionalLight, Model3D, PointLight, Rigidbody2D,
    Rigidbody3D, SphereCollider3D, SpriteAnimation, SpriteRenderer, Tag, Text, TextAlignment,
    Transform,
};
use vibe_ecs::{Entity, Uuid, World};

use crate::error::SceneError;

/// The file format version, so a future format change can be detected.
pub const SCENE_VERSION: u32 = 1;

/// A serialized scene.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SceneFile {
    /// Format version.
    pub version: u32,
    /// The scene's entities, in file order.
    pub entities: Vec<EntityRecord>,
}

/// One entity's serialized form.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EntityRecord {
    /// Stable identity, so references survive a reload.
    pub uuid: String,
    /// Optional human-facing name.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// Component values, keyed by component name.
    ///
    /// Kept as raw YAML rather than [`ComponentValue`] so an unrecognised key
    /// is preserved and skipped instead of failing the whole file.
    #[serde(default)]
    pub components: std::collections::BTreeMap<String, serde_norway::Value>,
}

/// Serializable form of [`Transform`].
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct TransformValue {
    pub x: f32,
    pub y: f32,
    #[serde(default)]
    pub z: f32,
    #[serde(default)]
    pub rotation: [f32; 4],
    #[serde(default = "one3")]
    pub scale: [f32; 3],
}

fn one3() -> [f32; 3] {
    [1.0, 1.0, 1.0]
}

/// Serializable form of [`Camera`].
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CameraValue {
    #[serde(default = "default_fov")]
    pub fov_y_radians: f32,
    #[serde(default = "default_near")]
    pub near: f32,
    #[serde(default = "default_far")]
    pub far: f32,
    #[serde(default = "default_clear")]
    pub clear_color: [f32; 4],
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub orthographic: Option<[f32; 2]>,
}

fn default_fov() -> f32 {
    std::f32::consts::FRAC_PI_3
}
fn default_near() -> f32 {
    0.1
}
fn default_far() -> f32 {
    1000.0
}
fn default_clear() -> [f32; 4] {
    [0.02, 0.02, 0.03, 1.0]
}

/// Serializable form of [`SpriteRenderer`].
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SpriteValue {
    #[serde(default)]
    pub texture_index: u32,
    #[serde(default)]
    pub offset: [f32; 2],
    #[serde(default = "default_sprite_size")]
    pub size: [f32; 2],
    #[serde(default)]
    pub rotation_radians: f32,
    #[serde(default = "one4")]
    pub tint: [f32; 4],
    #[serde(default)]
    pub uv_origin: [f32; 2],
    #[serde(default = "one2")]
    pub uv_size: [f32; 2],
    #[serde(default)]
    pub sort_order: i32,
}

fn default_sprite_size() -> [f32; 2] {
    [64.0, 64.0]
}
fn one2() -> [f32; 2] {
    [1.0, 1.0]
}
fn one4() -> [f32; 4] {
    [1.0, 1.0, 1.0, 1.0]
}

/// Serializable form of [`Text`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TextValue {
    #[serde(default)]
    pub content: String,
    #[serde(default = "default_text_size")]
    pub size: f32,
    #[serde(default = "one4")]
    pub color: [f32; 4],
    #[serde(default)]
    pub alignment: String,
    #[serde(default = "default_line_spacing")]
    pub line_spacing: f32,
}

fn default_text_size() -> f32 {
    16.0
}
fn default_line_spacing() -> f32 {
    1.2
}

/// Serializable form of [`SpriteAnimation`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AnimationValue {
    #[serde(default)]
    pub start_frame: u32,
    #[serde(default = "one_u32")]
    pub frame_count: u32,
    #[serde(default = "one_u32")]
    pub columns: u32,
    #[serde(default = "one2")]
    pub frame_uv_size: [f32; 2],
    #[serde(default = "default_frame_duration")]
    pub frame_duration: f32,
    #[serde(default)]
    pub loop_mode: String,
}

fn one_u32() -> u32 {
    1
}
fn default_frame_duration() -> f32 {
    1.0 / 12.0
}

/// Serializable form of [`Model3D`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelValue {
    #[serde(default)]
    pub mesh_path: String,
    #[serde(default)]
    pub skin_index: u32,
    #[serde(default = "one4")]
    pub tint: [f32; 4],
    #[serde(default)]
    pub sort_order: i32,
}

/// Serializable form of [`Animator`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AnimatorValue {
    #[serde(default)]
    pub clip_index: u32,
    #[serde(default)]
    pub time_seconds: f32,
    #[serde(default = "one_f32")]
    pub speed: f32,
    #[serde(default)]
    pub loop_mode: String,
    #[serde(default = "one_f32")]
    pub blend: f32,
}

fn one_f32() -> f32 {
    1.0
}

/// Serializable form of [`DirectionalLight`] and [`PointLight`].
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct LightValue {
    #[serde(default = "one3")]
    pub color: [f32; 3],
    #[serde(default = "one_f32")]
    pub intensity: f32,
    #[serde(default)]
    pub direction: [f32; 3],
    #[serde(default = "default_range")]
    pub range: f32,
}

fn default_range() -> f32 {
    10.0
}

/// Serializable form of [`Rigidbody2D`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Rigidbody2DValue {
    #[serde(default)]
    pub body_type: String,
    #[serde(default = "one_f32")]
    pub mass: f32,
    #[serde(default)]
    pub linear_damping: f32,
    #[serde(default)]
    pub angular_damping: f32,
    #[serde(default = "one_f32")]
    pub gravity_scale: f32,
    #[serde(default)]
    pub is_sensor: bool,
    #[serde(default)]
    pub is_rotatable: bool,
}

/// Serializable form of [`Rigidbody3D`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Rigidbody3DValue {
    #[serde(default)]
    pub body_type: String,
    #[serde(default = "one_f32")]
    pub mass: f32,
    #[serde(default)]
    pub linear_damping: f32,
    #[serde(default)]
    pub angular_damping: f32,
    #[serde(default = "one_f32")]
    pub gravity_scale: f32,
    #[serde(default)]
    pub is_sensor: bool,
    #[serde(default)]
    pub lock_linear: [bool; 3],
    #[serde(default)]
    pub lock_angular: [bool; 3],
}

/// Serializable form of [`BoxCollider2D`].
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Collider2DValue {
    #[serde(default = "one2")]
    pub size: [f32; 2],
    #[serde(default)]
    pub offset: [f32; 2],
    #[serde(default = "default_density")]
    pub density: f32,
    #[serde(default = "default_friction")]
    pub friction: f32,
    #[serde(default)]
    pub restitution: f32,
}

fn default_density() -> f32 {
    1.0
}
fn default_friction() -> f32 {
    0.3
}

/// Serializable form of [`BoxCollider3D`].
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Collider3DValue {
    #[serde(default = "one3")]
    pub size: [f32; 3],
    #[serde(default)]
    pub offset: [f32; 3],
    #[serde(default = "default_density")]
    pub density: f32,
    #[serde(default = "default_friction")]
    pub friction: f32,
    #[serde(default)]
    pub restitution: f32,
}

/// Serializable form of [`SphereCollider3D`].
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SphereValue {
    #[serde(default)]
    pub radius: f32,
    #[serde(default)]
    pub offset: [f32; 3],
    #[serde(default = "default_density")]
    pub density: f32,
    #[serde(default = "default_friction")]
    pub friction: f32,
    #[serde(default)]
    pub restitution: f32,
}

/// Serializable form of [`CapsuleCollider3D`].
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CapsuleValue {
    #[serde(default = "default_capsule_radius")]
    pub radius: f32,
    #[serde(default = "one_f32")]
    pub height: f32,
    #[serde(default)]
    pub offset: [f32; 3],
    #[serde(default)]
    pub axis: u8,
    #[serde(default = "default_density")]
    pub density: f32,
    #[serde(default = "default_friction")]
    pub friction: f32,
    #[serde(default)]
    pub restitution: f32,
}

fn default_capsule_radius() -> f32 {
    0.5
}

impl TransformValue {
    fn from_transform(t: &Transform) -> TransformValue {
        TransformValue {
            x: t.x,
            y: t.y,
            z: t.z,
            rotation: [t.rotation.x, t.rotation.y, t.rotation.z, t.rotation.w],
            scale: [t.scale.x, t.scale.y, t.scale.z],
        }
    }

    fn to_transform(self) -> Transform {
        Transform {
            x: self.x,
            y: self.y,
            z: self.z,
            rotation: Quat::from_xyzw(
                self.rotation[0],
                self.rotation[1],
                self.rotation[2],
                self.rotation[3],
            ),
            scale: Vec3::from_array(self.scale),
        }
    }
}

impl SpriteValue {
    fn from_sprite(s: &SpriteRenderer) -> SpriteValue {
        SpriteValue {
            texture_index: s.texture_index,
            offset: s.offset.to_array(),
            size: s.size.to_array(),
            rotation_radians: s.rotation_radians,
            tint: s.tint.to_array(),
            uv_origin: s.uv_origin.to_array(),
            uv_size: s.uv_size.to_array(),
            sort_order: s.sort_order,
        }
    }

    fn to_sprite(self) -> SpriteRenderer {
        SpriteRenderer {
            texture_index: self.texture_index,
            offset: Vec2::from_array(self.offset),
            size: Vec2::from_array(self.size),
            rotation_radians: self.rotation_radians,
            tint: Vec4::from_array(self.tint),
            uv_origin: Vec2::from_array(self.uv_origin),
            uv_size: Vec2::from_array(self.uv_size),
            sort_order: self.sort_order,
        }
    }
}

impl LightValue {
    fn from_directional(l: &DirectionalLight) -> LightValue {
        LightValue {
            color: l.color.to_array(),
            intensity: l.intensity,
            direction: l.direction.to_array(),
            range: 0.0,
        }
    }

    fn to_directional(self) -> DirectionalLight {
        DirectionalLight {
            color: Vec3::from_array(self.color),
            intensity: self.intensity,
            direction: Vec3::from_array(self.direction),
        }
    }

    fn from_point(l: &PointLight) -> LightValue {
        LightValue {
            color: l.color.to_array(),
            intensity: l.intensity,
            direction: [0.0; 3],
            range: l.range,
        }
    }

    fn to_point(self) -> PointLight {
        PointLight {
            color: Vec3::from_array(self.color),
            intensity: self.intensity,
            range: self.range,
            radius: 0.1,
        }
    }
}

fn to_transform_value(t: &Transform) -> TransformValue {
    TransformValue::from_transform(t)
}
fn from_transform_value(v: TransformValue) -> Transform {
    v.to_transform()
}
fn to_camera_value(c: &Camera) -> CameraValue {
    CameraValue {
        fov_y_radians: c.fov_y_radians,
        near: c.near,
        far: c.far,
        clear_color: c.clear_color.to_array(),
        orthographic: c.orthographic.map(|o| o.to_array()),
    }
}
fn from_camera_value(v: CameraValue) -> Camera {
    Camera {
        fov_y_radians: v.fov_y_radians,
        near: v.near,
        far: v.far,
        clear_color: Vec4::from_array(v.clear_color),
        orthographic: v.orthographic.map(Vec2::from_array),
    }
}
fn to_text_value(t: &Text) -> TextValue {
    TextValue {
        content: t.content.clone(),
        size: t.size,
        color: t.color.to_array(),
        alignment: format!("{:?}", t.alignment).to_lowercase(),
        line_spacing: t.line_spacing,
    }
}
fn from_text_value(v: TextValue) -> Text {
    Text {
        content: v.content,
        size: v.size,
        color: Vec4::from_array(v.color),
        alignment: match v.alignment.as_str() {
            "center" => TextAlignment::Center,
            "right" => TextAlignment::Right,
            _ => TextAlignment::Left,
        },
        line_spacing: v.line_spacing,
    }
}
fn to_animation_value(a: &SpriteAnimation) -> AnimationValue {
    AnimationValue {
        start_frame: a.start_frame,
        frame_count: a.frame_count,
        columns: a.columns,
        frame_uv_size: a.frame_uv_size.to_array(),
        frame_duration: a.frame_duration,
        loop_mode: format!("{:?}", a.loop_mode).to_lowercase(),
    }
}
fn from_animation_value(v: AnimationValue) -> SpriteAnimation {
    SpriteAnimation {
        start_frame: v.start_frame,
        frame_count: v.frame_count,
        columns: v.columns,
        frame_uv_size: Vec2::from_array(v.frame_uv_size),
        frame_duration: v.frame_duration,
        loop_mode: parse_loop(&v.loop_mode),
        current_frame: 0.0,
    }
}
fn parse_loop(s: &str) -> AnimationLoop {
    match s {
        "clamp" => AnimationLoop::Clamp,
        "pingpong" => AnimationLoop::PingPong,
        _ => AnimationLoop::Loop,
    }
}
fn to_model_value(m: &Model3D) -> ModelValue {
    ModelValue {
        mesh_path: m.mesh_path.clone(),
        skin_index: m.skin_index,
        tint: m.tint.to_array(),
        sort_order: m.sort_order,
    }
}
fn from_model_value(v: ModelValue) -> Model3D {
    Model3D {
        mesh_path: v.mesh_path,
        skin_index: v.skin_index,
        tint: Vec4::from_array(v.tint),
        sort_order: v.sort_order,
    }
}
fn to_animator_value(a: &Animator) -> AnimatorValue {
    AnimatorValue {
        clip_index: a.clip_index,
        time_seconds: a.time_seconds,
        speed: a.speed,
        loop_mode: format!("{:?}", a.loop_mode).to_lowercase(),
        blend: a.blend,
    }
}
fn from_animator_value(v: AnimatorValue) -> Animator {
    Animator {
        clip_index: v.clip_index,
        time_seconds: v.time_seconds,
        speed: v.speed,
        loop_mode: parse_loop(&v.loop_mode),
        blend: v.blend,
    }
}
fn to_rb2d(r: &Rigidbody2D) -> Rigidbody2DValue {
    Rigidbody2DValue {
        body_type: format!("{:?}", r.body_type).to_lowercase(),
        mass: r.mass,
        linear_damping: r.linear_damping,
        angular_damping: r.angular_damping,
        gravity_scale: r.gravity_scale,
        is_sensor: r.is_sensor,
        is_rotatable: r.is_rotatable,
    }
}
fn from_rb2d(v: Rigidbody2DValue) -> Rigidbody2D {
    Rigidbody2D {
        body_type: match v.body_type.as_str() {
            "kinematic" => BodyType2D::Kinematic,
            "dynamic" => BodyType2D::Dynamic,
            _ => BodyType2D::Static,
        },
        mass: v.mass,
        linear_damping: v.linear_damping,
        angular_damping: v.angular_damping,
        gravity_scale: v.gravity_scale,
        is_sensor: v.is_sensor,
        is_rotatable: v.is_rotatable,
    }
}
fn to_rb3d(r: &Rigidbody3D) -> Rigidbody3DValue {
    Rigidbody3DValue {
        body_type: format!("{:?}", r.body_type).to_lowercase(),
        mass: r.mass,
        linear_damping: r.linear_damping,
        angular_damping: r.angular_damping,
        gravity_scale: r.gravity_scale,
        is_sensor: r.is_sensor,
        lock_linear: [r.lock_linear.0, r.lock_linear.1, r.lock_linear.2],
        lock_angular: [r.lock_angular.0, r.lock_angular.1, r.lock_angular.2],
    }
}
fn from_rb3d(v: Rigidbody3DValue) -> Rigidbody3D {
    Rigidbody3D {
        body_type: match v.body_type.as_str() {
            "kinematic" => BodyType3D::Kinematic,
            "dynamic" => BodyType3D::Dynamic,
            _ => BodyType3D::Static,
        },
        mass: v.mass,
        linear_damping: v.linear_damping,
        angular_damping: v.angular_damping,
        gravity_scale: v.gravity_scale,
        is_sensor: v.is_sensor,
        lock_linear: (v.lock_linear[0], v.lock_linear[1], v.lock_linear[2]),
        lock_angular: (v.lock_angular[0], v.lock_angular[1], v.lock_angular[2]),
    }
}
fn to_box2d(c: &BoxCollider2D) -> Collider2DValue {
    Collider2DValue {
        size: c.size.to_array(),
        offset: c.offset.to_array(),
        density: c.material.density,
        friction: c.material.friction,
        restitution: c.material.restitution,
    }
}
fn from_box2d(v: Collider2DValue) -> BoxCollider2D {
    BoxCollider2D {
        size: Vec2::from_array(v.size),
        offset: Vec2::from_array(v.offset),
        material: ColliderMaterial {
            density: v.density,
            friction: v.friction,
            restitution: v.restitution,
        },
    }
}
fn to_box3d(c: &BoxCollider3D) -> Collider3DValue {
    Collider3DValue {
        size: c.size.to_array(),
        offset: c.offset.to_array(),
        density: c.material.density,
        friction: c.material.friction,
        restitution: c.material.restitution,
    }
}
fn from_box3d(v: Collider3DValue) -> BoxCollider3D {
    BoxCollider3D {
        size: Vec3::from_array(v.size),
        offset: Vec3::from_array(v.offset),
        material: ColliderMaterial {
            density: v.density,
            friction: v.friction,
            restitution: v.restitution,
        },
    }
}
fn to_sphere(c: &SphereCollider3D) -> SphereValue {
    SphereValue {
        radius: c.radius,
        offset: c.offset.to_array(),
        density: c.material.density,
        friction: c.material.friction,
        restitution: c.material.restitution,
    }
}
fn from_sphere(v: SphereValue) -> SphereCollider3D {
    SphereCollider3D {
        radius: v.radius,
        offset: Vec3::from_array(v.offset),
        material: ColliderMaterial {
            density: v.density,
            friction: v.friction,
            restitution: v.restitution,
        },
    }
}
fn to_capsule(c: &CapsuleCollider3D) -> CapsuleValue {
    CapsuleValue {
        radius: c.radius,
        height: c.height,
        offset: c.offset.to_array(),
        axis: c.axis,
        density: c.material.density,
        friction: c.material.friction,
        restitution: c.material.restitution,
    }
}
fn from_capsule(v: CapsuleValue) -> CapsuleCollider3D {
    CapsuleCollider3D {
        radius: v.radius,
        height: v.height,
        offset: Vec3::from_array(v.offset),
        axis: v.axis,
        material: ColliderMaterial {
            density: v.density,
            friction: v.friction,
            restitution: v.restitution,
        },
    }
}

fn to_sprite_value(s: &SpriteRenderer) -> SpriteValue {
    SpriteValue::from_sprite(s)
}
fn from_sprite_value(v: SpriteValue) -> SpriteRenderer {
    v.to_sprite()
}
fn to_directional_value(l: &DirectionalLight) -> LightValue {
    LightValue::from_directional(l)
}
fn from_directional_value(v: LightValue) -> DirectionalLight {
    v.to_directional()
}
fn to_point_value(l: &PointLight) -> LightValue {
    LightValue::from_point(l)
}
fn from_point_value(v: LightValue) -> PointLight {
    v.to_point()
}

/// Serialize a world to a scene file.
pub fn world_to_scene(world: &World) -> SceneFile {
    let mut entities = Vec::new();
    for entity in world.entities() {
        let uuid = world.uuid(entity).unwrap_or(Uuid::NIL);
        let mut components = std::collections::BTreeMap::new();
        let mut name = None;

        if let Some(t) = world.get::<Tag>(entity) {
            name = Some(t.0.clone());
        }
        if let Some(v) = world.get::<Transform>(entity) {
            components.insert("Transform".to_string(), to_raw(&to_transform_value(v)));
        }
        if let Some(v) = world.get::<Camera>(entity) {
            components.insert("Camera".to_string(), to_raw(&to_camera_value(v)));
        }
        if let Some(v) = world.get::<SpriteRenderer>(entity) {
            components.insert("SpriteRenderer".to_string(), to_raw(&to_sprite_value(v)));
        }
        if let Some(v) = world.get::<Text>(entity) {
            components.insert("Text".to_string(), to_raw(&to_text_value(v)));
        }
        if let Some(v) = world.get::<SpriteAnimation>(entity) {
            components.insert(
                "SpriteAnimation".to_string(),
                to_raw(&to_animation_value(v)),
            );
        }
        if let Some(v) = world.get::<Model3D>(entity) {
            components.insert("Model3D".to_string(), to_raw(&to_model_value(v)));
        }
        if let Some(v) = world.get::<Animator>(entity) {
            components.insert("Animator".to_string(), to_raw(&to_animator_value(v)));
        }
        if let Some(v) = world.get::<DirectionalLight>(entity) {
            components.insert(
                "DirectionalLight".to_string(),
                to_raw(&to_directional_value(v)),
            );
        }
        if let Some(v) = world.get::<PointLight>(entity) {
            components.insert("PointLight".to_string(), to_raw(&to_point_value(v)));
        }
        if let Some(v) = world.get::<Rigidbody2D>(entity) {
            components.insert("Rigidbody2D".to_string(), to_raw(&to_rb2d(v)));
        }
        if let Some(v) = world.get::<Rigidbody3D>(entity) {
            components.insert("Rigidbody3D".to_string(), to_raw(&to_rb3d(v)));
        }
        if let Some(v) = world.get::<BoxCollider2D>(entity) {
            components.insert("BoxCollider2D".to_string(), to_raw(&to_box2d(v)));
        }
        if let Some(v) = world.get::<BoxCollider3D>(entity) {
            components.insert("BoxCollider3D".to_string(), to_raw(&to_box3d(v)));
        }
        if let Some(v) = world.get::<SphereCollider3D>(entity) {
            components.insert("SphereCollider3D".to_string(), to_raw(&to_sphere(v)));
        }
        if let Some(v) = world.get::<CapsuleCollider3D>(entity) {
            components.insert("CapsuleCollider3D".to_string(), to_raw(&to_capsule(v)));
        }

        entities.push(EntityRecord {
            uuid: uuid.to_string(),
            name,
            components,
        });
    }
    SceneFile {
        version: SCENE_VERSION,
        entities,
    }
}

/// Load a scene file into a world.
///
/// Entities are created in file order and keep their UUIDs, so a save/load
/// round trip is stable.
pub fn scene_to_world(scene: &SceneFile) -> Result<World, SceneError> {
    if scene.version > SCENE_VERSION {
        return Err(SceneError::UnsupportedVersion {
            found: scene.version,
            supported: SCENE_VERSION,
        });
    }
    let mut world = World::new();
    for record in &scene.entities {
        let uuid = Uuid::parse(&record.uuid).ok_or_else(|| SceneError::BadUuid {
            uuid: record.uuid.clone(),
        })?;
        let entity = world.spawn_with_uuid(uuid);

        for (key, value) in &record.components {
            apply_component(&mut world, entity, key, value);
        }
        if let Some(name) = &record.name {
            world.add(entity, Tag(name.clone()));
        }
    }
    Ok(world)
}

/// Turn a component into the raw YAML value stored in the scene file.
fn to_raw<T: Serialize>(value: &T) -> serde_norway::Value {
    serde_norway::to_value(value).expect("component values are always serializable")
}

/// Attach one component from its raw YAML value.
///
/// The map key names the component, so the value carries no tag of its own. A
/// key this build does not know, or one whose value does not fit the component,
/// is skipped: a scene written by a newer build still loads, minus the parts
/// this build cannot represent.
fn apply_component(world: &mut World, entity: Entity, key: &str, raw: &serde_norway::Value) {
    macro_rules! decode {
        ($ty:ty, $body:expr) => {
            if let Ok(v) = serde_norway::from_value::<$ty>(raw.clone()) {
                world.add(entity, $body(v));
            }
        };
    }

    match key {
        "Transform" => decode!(TransformValue, from_transform_value),
        "Camera" => decode!(CameraValue, from_camera_value),
        "SpriteRenderer" => decode!(SpriteValue, from_sprite_value),
        "Text" => decode!(TextValue, from_text_value),
        "SpriteAnimation" => decode!(AnimationValue, from_animation_value),
        "Model3D" => decode!(ModelValue, from_model_value),
        "Animator" => decode!(AnimatorValue, from_animator_value),
        "DirectionalLight" => decode!(LightValue, from_directional_value),
        "PointLight" => decode!(LightValue, from_point_value),
        "Rigidbody2D" => decode!(Rigidbody2DValue, from_rb2d),
        "Rigidbody3D" => decode!(Rigidbody3DValue, from_rb3d),
        "BoxCollider2D" => decode!(Collider2DValue, from_box2d),
        "BoxCollider3D" => decode!(Collider3DValue, from_box3d),
        "SphereCollider3D" => decode!(SphereValue, from_sphere),
        "CapsuleCollider3D" => decode!(CapsuleValue, from_capsule),
        _ => {}
    }
}

/// Render a world as YAML.
///
/// `serde_norway` has no `to_string_pretty`, but its `Serializer` emits the
/// indented form, which is the point of a format people hand-edit.
pub fn world_to_yaml(world: &World) -> String {
    let scene = world_to_scene(world);
    let mut out: Vec<u8> = Vec::new();
    let mut serializer = serde_norway::Serializer::new(&mut out);
    scene
        .serialize(&mut serializer)
        .expect("scene is always serializable");
    String::from_utf8(out).expect("the yaml serializer only emits utf-8")
}

/// Parse YAML into a world.
pub fn yaml_to_world(yaml: &str) -> Result<World, SceneError> {
    let scene: SceneFile =
        serde_norway::from_str(yaml).map_err(|e| SceneError::Parse(e.to_string()))?;
    scene_to_world(&scene)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_world() -> World {
        let mut w = World::new();
        let e = w.spawn();
        w.add(e, Tag::new("player"));
        w.add(
            e,
            Transform {
                x: 10.0,
                y: 20.0,
                ..Default::default()
            },
        );
        w.add(
            e,
            SpriteRenderer {
                texture_index: 3,
                size: Vec2::splat(32.0),
                ..Default::default()
            },
        );
        w.add(
            e,
            Rigidbody2D {
                body_type: BodyType2D::Dynamic,
                mass: 2.0,
                ..Default::default()
            },
        );
        w
    }

    #[test]
    fn round_trip_preserves_entity_count() {
        let w = sample_world();
        let yaml = world_to_yaml(&w);
        let back = yaml_to_world(&yaml).unwrap();
        assert_eq!(back.len(), 1);
    }

    #[test]
    fn round_trip_preserves_transform() {
        let w = sample_world();
        let back = yaml_to_world(&world_to_yaml(&w)).unwrap();
        let e = back.entities().next().unwrap();
        let t = back.get::<Transform>(e).unwrap();
        assert_eq!(t.x, 10.0);
        assert_eq!(t.y, 20.0);
    }

    #[test]
    fn round_trip_preserves_tag_as_name() {
        let w = sample_world();
        let back = yaml_to_world(&world_to_yaml(&w)).unwrap();
        let e = back.entities().next().unwrap();
        assert_eq!(
            back.get::<Tag>(e).map(|t| t.0.clone()),
            Some("player".to_string())
        );
    }

    #[test]
    fn round_trip_preserves_sprite() {
        let w = sample_world();
        let back = yaml_to_world(&world_to_yaml(&w)).unwrap();
        let e = back.entities().next().unwrap();
        let s = back.get::<SpriteRenderer>(e).unwrap();
        assert_eq!(s.texture_index, 3);
        assert_eq!(s.size, Vec2::splat(32.0));
    }

    #[test]
    fn round_trip_preserves_rigidbody_body_type() {
        let w = sample_world();
        let back = yaml_to_world(&world_to_yaml(&w)).unwrap();
        let e = back.entities().next().unwrap();
        let r = back.get::<Rigidbody2D>(e).unwrap();
        assert_eq!(r.body_type, BodyType2D::Dynamic);
        assert_eq!(r.mass, 2.0);
    }

    #[test]
    fn uuids_survive_a_round_trip() {
        let w = sample_world();
        let before: Vec<String> = w
            .entities()
            .map(|e| w.uuid(e).unwrap().to_string())
            .collect();
        let back = yaml_to_world(&world_to_yaml(&w)).unwrap();
        let after: Vec<String> = back
            .entities()
            .map(|e| back.uuid(e).unwrap().to_string())
            .collect();
        assert_eq!(before, after);
    }

    #[test]
    fn default_transform_scale_survives() {
        let w = sample_world();
        let back = yaml_to_world(&world_to_yaml(&w)).unwrap();
        let e = back.entities().next().unwrap();
        assert_eq!(back.get::<Transform>(e).unwrap().scale, Vec3::ONE);
    }

    #[test]
    fn default_rotation_survives() {
        let w = sample_world();
        let back = yaml_to_world(&world_to_yaml(&w)).unwrap();
        let e = back.entities().next().unwrap();
        assert_eq!(back.get::<Transform>(e).unwrap().rotation, Quat::IDENTITY);
    }

    #[test]
    fn empty_world_round_trips() {
        let w = World::new();
        let back = yaml_to_world(&world_to_yaml(&w)).unwrap();
        assert!(back.is_empty());
    }

    #[test]
    fn multiple_entities_round_trip() {
        let mut w = World::new();
        for i in 0..5 {
            let e = w.spawn();
            w.add(e, Tag::new(format!("e{i}")));
            w.add(
                e,
                Transform {
                    x: i as f32,
                    ..Default::default()
                },
            );
        }
        let back = yaml_to_world(&world_to_yaml(&w)).unwrap();
        assert_eq!(back.len(), 5);
        assert_eq!(back.count::<Tag>(), 5);
    }

    #[test]
    fn camera_round_trips_with_orthographic() {
        let mut w = World::new();
        let e = w.spawn();
        w.add(
            e,
            Camera {
                orthographic: Some(Vec2::splat(10.0)),
                ..Default::default()
            },
        );
        let back = yaml_to_world(&world_to_yaml(&w)).unwrap();
        let e2 = back.entities().next().unwrap();
        assert_eq!(
            back.get::<Camera>(e2).unwrap().orthographic,
            Some(Vec2::splat(10.0))
        );
    }

    #[test]
    fn text_alignment_round_trips() {
        let mut w = World::new();
        let e = w.spawn();
        w.add(
            e,
            Text {
                content: "hi".into(),
                alignment: TextAlignment::Center,
                ..Default::default()
            },
        );
        let back = yaml_to_world(&world_to_yaml(&w)).unwrap();
        let e2 = back.entities().next().unwrap();
        assert_eq!(
            back.get::<Text>(e2).unwrap().alignment,
            TextAlignment::Center
        );
    }

    #[test]
    fn animation_loop_mode_round_trips() {
        let mut w = World::new();
        let e = w.spawn();
        w.add(
            e,
            SpriteAnimation {
                loop_mode: AnimationLoop::Clamp,
                ..Default::default()
            },
        );
        let back = yaml_to_world(&world_to_yaml(&w)).unwrap();
        let e2 = back.entities().next().unwrap();
        assert_eq!(
            back.get::<SpriteAnimation>(e2).unwrap().loop_mode,
            AnimationLoop::Clamp
        );
    }

    #[test]
    fn collider_material_round_trips() {
        let mut w = World::new();
        let e = w.spawn();
        w.add(
            e,
            BoxCollider3D {
                size: Vec3::splat(2.0),
                material: ColliderMaterial {
                    density: 3.0,
                    friction: 0.8,
                    restitution: 0.5,
                },
                ..Default::default()
            },
        );
        let back = yaml_to_world(&world_to_yaml(&w)).unwrap();
        let e2 = back.entities().next().unwrap();
        let c = back.get::<BoxCollider3D>(e2).unwrap();
        assert_eq!(c.material.density, 3.0);
        assert_eq!(c.material.friction, 0.8);
        assert_eq!(c.material.restitution, 0.5);
    }

    #[test]
    fn capsule_round_trips() {
        let mut w = World::new();
        let e = w.spawn();
        w.add(
            e,
            CapsuleCollider3D {
                radius: 0.7,
                height: 2.0,
                axis: 1,
                ..Default::default()
            },
        );
        let back = yaml_to_world(&world_to_yaml(&w)).unwrap();
        let e2 = back.entities().next().unwrap();
        let c = back.get::<CapsuleCollider3D>(e2).unwrap();
        assert_eq!(c.radius, 0.7);
        assert_eq!(c.axis, 1);
    }

    #[test]
    fn rigidbody3d_locks_round_trip() {
        let mut w = World::new();
        let e = w.spawn();
        w.add(
            e,
            Rigidbody3D {
                body_type: BodyType3D::Kinematic,
                lock_linear: (true, false, true),
                ..Default::default()
            },
        );
        let back = yaml_to_world(&world_to_yaml(&w)).unwrap();
        let e2 = back.entities().next().unwrap();
        let r = back.get::<Rigidbody3D>(e2).unwrap();
        assert_eq!(r.body_type, BodyType3D::Kinematic);
        assert_eq!(r.lock_linear, (true, false, true));
    }

    #[test]
    fn all_light_types_round_trip() {
        let mut w = World::new();
        let e = w.spawn();
        w.add(
            e,
            DirectionalLight {
                intensity: 2.0,
                ..Default::default()
            },
        );
        w.add(
            e,
            PointLight {
                range: 25.0,
                ..Default::default()
            },
        );
        let back = yaml_to_world(&world_to_yaml(&w)).unwrap();
        let e2 = back.entities().next().unwrap();
        assert_eq!(back.get::<DirectionalLight>(e2).unwrap().intensity, 2.0);
        assert_eq!(back.get::<PointLight>(e2).unwrap().range, 25.0);
    }

    #[test]
    fn model_and_animator_round_trip() {
        let mut w = World::new();
        let e = w.spawn();
        w.add(
            e,
            Model3D {
                mesh_path: "hero.glb".into(),
                skin_index: 1,
                ..Default::default()
            },
        );
        w.add(
            e,
            Animator {
                clip_index: 2,
                speed: 1.5,
                ..Default::default()
            },
        );
        let back = yaml_to_world(&world_to_yaml(&w)).unwrap();
        let e2 = back.entities().next().unwrap();
        assert_eq!(back.get::<Model3D>(e2).unwrap().mesh_path, "hero.glb");
        assert_eq!(back.get::<Animator>(e2).unwrap().clip_index, 2);
    }

    #[test]
    fn yaml_is_human_readable() {
        let w = sample_world();
        let yaml = world_to_yaml(&w);
        assert!(yaml.contains("version:"), "{yaml}");
        assert!(yaml.contains("entities:"), "{yaml}");
        assert!(yaml.contains("Transform"), "{yaml}");
    }

    #[test]
    fn bad_yaml_is_a_parse_error() {
        let err = yaml_to_world("this: [is not: valid").unwrap_err();
        assert!(matches!(err, SceneError::Parse(_)));
    }

    #[test]
    fn bad_uuid_is_reported() {
        let yaml = "version: 1\nentities:\n  - uuid: not-a-uuid\n    components: {}\n";
        let err = yaml_to_world(yaml).unwrap_err();
        assert!(matches!(err, SceneError::BadUuid { .. }));
    }

    #[test]
    fn newer_version_is_rejected() {
        let yaml = "version: 99\nentities: []\n";
        let err = yaml_to_world(yaml).unwrap_err();
        assert!(matches!(
            err,
            SceneError::UnsupportedVersion { found: 99, .. }
        ));
    }

    #[test]
    fn unknown_component_key_is_ignored() {
        // Forward compatibility: an older build must not fail on a file written
        // by a newer one.
        let yaml = "version: 1\nentities:\n  - uuid: 9f1c2d3e-4a5b-4c6d-8e9f-0a1b2c3d4e5f\n    components:\n      FutureThing:\n        x: 1\n";
        let w = yaml_to_world(yaml).unwrap();
        assert_eq!(w.len(), 1);
    }

    #[test]
    fn entity_without_components_loads() {
        let yaml = "version: 1\nentities:\n  - uuid: 9f1c2d3e-4a5b-4c6d-8e9f-0a1b2c3d4e5f\n";
        let w = yaml_to_world(yaml).unwrap();
        assert_eq!(w.len(), 1);
    }
}
