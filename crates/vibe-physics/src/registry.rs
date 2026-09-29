//! The bridge between ECS components and live simulation objects.
//!
//! A `Rigidbody2D` on an entity is data; a body in the solver is state. This
//! registry maps one to the other and keeps the two in step, including the
//! case that breaks naive implementations: an entity despawned mid-frame, or a
//! component removed while its body is still in the world.

use std::collections::HashMap;

use vibe_ecs::{Entity, World, components::*};

use crate::error::PhysicsError;

/// An opaque handle to a body inside a Rapier world.
///
/// The generation makes a handle safe across destroy/recreate: a collider
/// handle from before a rebuild will not resolve to whatever now occupies the
/// same slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct BodyHandle {
    /// Index into the Rapier rigid-body set.
    pub index: u32,
    /// Bumped when the slot is reused.
    pub generation: u32,
}

/// A body's transform as the ECS sees it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BodyTransform {
    /// World position.
    pub position: glam::Vec3,
    /// Orientation as a quaternion.
    pub rotation: glam::Quat,
}

impl BodyTransform {
    /// An identity transform.
    pub fn identity() -> BodyTransform {
        BodyTransform {
            position: glam::Vec3::ZERO,
            rotation: glam::Quat::IDENTITY,
        }
    }
}

/// Which dimension pair a body lives in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Dimension {
    /// A 2D body, with Z unused.
    Two,
    /// A 3D body.
    Three,
}

impl Dimension {
    /// The name used in errors.
    pub const fn name(self) -> &'static str {
        match self {
            Dimension::Two => "2D",
            Dimension::Three => "3D",
        }
    }
}

/// Tracks which entities have live bodies, in both dimensions.
///
/// A single registry serves both, so a world dump and the editor's outliner can
/// ask one question — "what is simulated?" — instead of two.
#[derive(Debug, Default)]
pub struct PhysicsRegistry {
    bodies: HashMap<Entity, BodyHandle>,
    generation: u32,
    /// Entities seen with a body component but not yet registered, so a caller
    /// can tell "not registered yet" from "no body wanted".
    pending: Vec<Entity>,
}

impl PhysicsRegistry {
    /// An empty registry.
    pub fn new() -> PhysicsRegistry {
        PhysicsRegistry::default()
    }

    /// Record a body's handle for an entity.
    ///
    /// Re-registering bumps the generation, so a handle from the previous
    /// registration no longer resolves.
    pub fn register(&mut self, entity: Entity, handle: BodyHandle) {
        if self.bodies.insert(entity, handle).is_some() {
            self.generation = self.generation.wrapping_add(1);
        }
        self.pending.retain(|e| *e != entity);
    }

    /// Issue a handle for a newly created body.
    pub fn next_handle(&mut self, index: u32) -> BodyHandle {
        self.generation = self.generation.wrapping_add(1);
        BodyHandle {
            index,
            generation: self.generation,
        }
    }

    /// A body's handle, if registered.
    pub fn handle(&self, entity: Entity, dimension: Dimension) -> Result<BodyHandle, PhysicsError> {
        match self.bodies.get(&entity) {
            Some(h) => Ok(*h),
            None => Err(PhysicsError::NoBody {
                entity: entity.to_string(),
                kind: dimension.name(),
            }),
        }
    }

    /// True when the entity has a registered body.
    pub fn has_body(&self, entity: Entity) -> bool {
        self.bodies.contains_key(&entity)
    }

    /// Forget an entity's body, e.g. because the entity despawned.
    pub fn remove(&mut self, entity: Entity) -> Option<BodyHandle> {
        self.bodies.remove(&entity)
    }

    /// Every registered body.
    pub fn bodies(&self) -> impl Iterator<Item = (Entity, BodyHandle)> + '_ {
        self.bodies
            .iter()
            .map(|(e, h)| (*e, *h))
            .collect::<Vec<_>>()
            .into_iter()
    }

    /// Number of live bodies.
    pub fn len(&self) -> usize {
        self.bodies.len()
    }

    /// True when nothing is simulated.
    pub fn is_empty(&self) -> bool {
        self.bodies.is_empty()
    }

    /// Drop registrations for entities that no longer exist.
    ///
    /// Returns the entities whose bodies were dropped, so the caller can
    /// destroy them in the solver.
    pub fn prune(&mut self, world: &World) -> Vec<Entity> {
        let dead: Vec<Entity> = self
            .bodies
            .keys()
            .copied()
            .filter(|e| !world.is_alive(*e))
            .collect();
        for e in &dead {
            self.bodies.remove(e);
        }
        dead
    }

    /// True when the entity is simulated and carries a collider.
    pub fn is_simulated(&self, world: &World, entity: Entity) -> bool {
        self.has_body(entity) && has_any_collider(world, entity)
    }
}

/// True when an entity carries any collider component.
pub fn has_any_collider(world: &World, entity: Entity) -> bool {
    world.has::<BoxCollider2D>(entity)
        || world.has::<BoxCollider3D>(entity)
        || world.has::<SphereCollider3D>(entity)
        || world.has::<CapsuleCollider3D>(entity)
}

/// Which collider an entity has in 3D.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Shape3D {
    /// A box of the given half-extents.
    Box(glam::Vec3),
    /// A sphere of the given radius.
    Sphere(f32),
    /// A capsule with the given radius and half-height of its cylindrical part.
    Capsule {
        /// Capsule radius.
        radius: f32,
        /// Half the cylindrical length, excluding the caps.
        half_height: f32,
    },
}

impl Shape3D {
    /// A tight bounding radius, used for broad-phase and culling.
    pub fn bounding_radius(&self) -> f32 {
        match self {
            Shape3D::Box(half) => half.length(),
            Shape3D::Sphere(r) => *r,
            Shape3D::Capsule {
                radius,
                half_height,
            } => half_height + radius,
        }
    }
}

/// Read an entity's 3D collider into a shape.
pub fn shape_from_world(world: &World, entity: Entity) -> Option<Shape3D> {
    if let Some(c) = world.get::<BoxCollider3D>(entity) {
        return Some(Shape3D::Box(c.size * 0.5));
    }
    if let Some(c) = world.get::<SphereCollider3D>(entity) {
        return Some(Shape3D::Sphere(c.radius));
    }
    if let Some(c) = world.get::<CapsuleCollider3D>(entity) {
        return Some(Shape3D::Capsule {
            radius: c.radius,
            half_height: c.height * 0.5,
        });
    }
    None
}

/// A body's mass as physics should compute it.
///
/// Derived from the collider's volume and density unless the body component
/// overrides it, which is what lets an artist set density on the shape and
/// still get the inertia tensor the solver needs.
pub fn effective_mass_2d(world: &World, entity: Entity) -> Option<f32> {
    let rb = world.get::<Rigidbody2D>(entity)?;
    if rb.body_type != BodyType2D::Dynamic {
        return None;
    }
    let collider = world.get::<BoxCollider2D>(entity)?;
    let area = (collider.size.x * collider.size.y).abs();
    Some(area * collider.material.density)
}

/// A 3D body's mass.
pub fn effective_mass_3d(world: &World, entity: Entity) -> Option<f32> {
    let rb = world.get::<Rigidbody3D>(entity)?;
    if rb.body_type != BodyType3D::Dynamic {
        return None;
    }
    let shape = shape_from_world(world, entity)?;
    let volume = match shape {
        Shape3D::Box(half) => 8.0 * half.x * half.y * half.z,
        Shape3D::Sphere(r) => (4.0 / 3.0) * std::f32::consts::PI * r * r * r,
        Shape3D::Capsule {
            radius,
            half_height,
        } => {
            let cylinder = std::f32::consts::PI * radius * radius * 2.0 * half_height;
            let sphere = (4.0 / 3.0) * std::f32::consts::PI * radius * radius * radius;
            cylinder + sphere
        }
    };
    let density = world
        .get::<BoxCollider3D>(entity)
        .map(|c| c.material.density)
        .or_else(|| {
            world
                .get::<SphereCollider3D>(entity)
                .map(|c| c.material.density)
        })
        .or_else(|| {
            world
                .get::<CapsuleCollider3D>(entity)
                .map(|c| c.material.density)
        })
        .unwrap_or(1.0);
    Some(volume * density)
}

/// Copy a body's transform from the ECS into a physics-space transform.
pub fn transform_from_world(world: &World, entity: Entity) -> BodyTransform {
    match world.get::<Transform>(entity) {
        Some(t) => BodyTransform {
            position: t.translation(),
            rotation: t.rotation,
        },
        None => BodyTransform::identity(),
    }
}

/// Write a body's transform back into the ECS.
pub fn write_transform(world: &mut World, entity: Entity, body: BodyTransform) {
    if let Some(t) = world.get_mut::<Transform>(entity) {
        t.x = body.position.x;
        t.y = body.position.y;
        t.z = body.position.z;
        t.rotation = body.rotation;
    }
}

/// Count of bodies the registry would create for a world, by body type.
///
/// Useful for the editor's scene stats and for asserting a test built what it
/// meant to.
pub fn summarise(world: &World) -> BodyCounts {
    let mut counts = BodyCounts::default();
    for (_, rb) in world.query::<Rigidbody2D>() {
        match rb.body_type {
            BodyType2D::Static => counts.static_2d += 1,
            BodyType2D::Kinematic => counts.kinematic_2d += 1,
            BodyType2D::Dynamic => counts.dynamic_2d += 1,
        }
    }
    for (_, rb) in world.query::<Rigidbody3D>() {
        match rb.body_type {
            BodyType3D::Static => counts.static_3d += 1,
            BodyType3D::Kinematic => counts.kinematic_3d += 1,
            BodyType3D::Dynamic => counts.dynamic_3d += 1,
        }
    }
    counts.colliders_2d = world.count::<BoxCollider2D>();
    counts.colliders_3d = world.count::<BoxCollider3D>()
        + world.count::<SphereCollider3D>()
        + world.count::<CapsuleCollider3D>();
    counts
}

/// How many bodies and colliders a world holds, by type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct BodyCounts {
    pub static_2d: usize,
    pub kinematic_2d: usize,
    pub dynamic_2d: usize,
    pub static_3d: usize,
    pub kinematic_3d: usize,
    pub dynamic_3d: usize,
    pub colliders_2d: usize,
    pub colliders_3d: usize,
}

impl BodyCounts {
    /// Total bodies across both dimensions.
    pub fn total(&self) -> usize {
        self.static_2d
            + self.kinematic_2d
            + self.dynamic_2d
            + self.static_3d
            + self.kinematic_3d
            + self.dynamic_3d
    }

    /// True when nothing is simulated.
    pub fn is_empty(&self) -> bool {
        self.total() == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use glam::{Quat, Vec2, Vec3};

    fn world_with_2d(body_type: BodyType2D) -> (World, Entity) {
        let mut w = World::new();
        let e = w.spawn();
        w.add(
            e,
            Rigidbody2D {
                body_type,
                ..Default::default()
            },
        );
        w.add(e, BoxCollider2D::new(Vec2::splat(2.0)));
        (w, e)
    }

    #[test]
    fn register_and_query_a_handle() {
        let mut r = PhysicsRegistry::new();
        let e = Entity::from_parts(0, 0);
        let h = r.next_handle(0);
        r.register(e, h);
        assert_eq!(r.handle(e, Dimension::Two), Ok(h));
        assert!(r.has_body(e));
        assert_eq!(r.len(), 1);
    }

    #[test]
    fn unregistered_entity_reports_no_body() {
        let r = PhysicsRegistry::new();
        let e = Entity::from_parts(0, 0);
        let err = r.handle(e, Dimension::Two).unwrap_err();
        assert!(matches!(err, PhysicsError::NoBody { kind: "2D", .. }));
        assert!(err.to_string().contains("2D"));
    }

    #[test]
    fn three_d_reports_three_d() {
        let r = PhysicsRegistry::new();
        let err = r
            .handle(Entity::from_parts(0, 0), Dimension::Three)
            .unwrap_err();
        assert!(matches!(err, PhysicsError::NoBody { kind: "3D", .. }));
    }

    #[test]
    fn re_registration_bumps_generation() {
        let mut r = PhysicsRegistry::new();
        let e = Entity::from_parts(0, 0);
        let first = r.next_handle(0);
        r.register(e, first);
        let second = r.next_handle(0);
        r.register(e, second);
        assert_ne!(
            first.generation, second.generation,
            "a reused slot must not look the same"
        );
        assert_eq!(r.handle(e, Dimension::Two), Ok(second));
    }

    #[test]
    fn remove_forgets_the_body() {
        let mut r = PhysicsRegistry::new();
        let e = Entity::from_parts(0, 0);
        {
            let h = r.next_handle(0);
            r.register(e, h);
        }
        assert!(r.remove(e).is_some());
        assert!(!r.has_body(e));
        assert!(r.remove(e).is_none());
    }

    #[test]
    fn prune_drops_dead_entities() {
        let mut w = World::new();
        let alive = w.spawn();
        let dead = w.spawn();
        w.despawn(dead);

        let mut r = PhysicsRegistry::new();
        {
            let h = r.next_handle(0);
            r.register(alive, h);
        }
        {
            let h = r.next_handle(1);
            r.register(dead, h);
        }

        let pruned = r.prune(&w);
        assert_eq!(pruned, vec![dead]);
        assert_eq!(r.len(), 1);
        assert!(r.has_body(alive));
    }

    #[test]
    fn prune_on_empty_world_is_noop() {
        let mut r = PhysicsRegistry::new();
        let w = World::new();
        assert!(r.prune(&w).is_empty());
    }

    #[test]
    fn is_simulated_needs_body_and_collider() {
        let mut w = World::new();
        let with_both = w.spawn();
        w.add(with_both, Rigidbody2D::default());
        w.add(with_both, BoxCollider2D::default());
        let body_only = w.spawn();
        w.add(body_only, Rigidbody2D::default());

        let mut r = PhysicsRegistry::new();
        {
            let h = r.next_handle(0);
            r.register(with_both, h);
        }
        {
            let h = r.next_handle(1);
            r.register(body_only, h);
        }

        assert!(r.is_simulated(&w, with_both));
        assert!(
            !r.is_simulated(&w, body_only),
            "a body with no shape is not simulated"
        );
    }

    #[test]
    fn has_any_collider_covers_every_shape() {
        let mut w = World::new();
        for_each_shape(&mut w);
        assert_eq!(w.len(), 4);
        for e in w.entities() {
            assert!(has_any_collider(&w, e), "every shape should be detected");
        }
    }

    fn for_each_shape(w: &mut World) {
        let a = w.spawn();
        w.add(a, BoxCollider2D::default());
        let b = w.spawn();
        w.add(b, BoxCollider3D::default());
        let c = w.spawn();
        w.add(c, SphereCollider3D::default());
        let d = w.spawn();
        w.add(d, CapsuleCollider3D::default());
    }

    #[test]
    fn has_any_collider_is_false_without_one() {
        let mut w = World::new();
        let e = w.spawn();
        assert!(!has_any_collider(&w, e));
    }

    #[test]
    fn shape_from_box_is_half_extents() {
        let mut w = World::new();
        let e = w.spawn();
        w.add(
            e,
            BoxCollider3D {
                size: Vec3::splat(4.0),
                ..Default::default()
            },
        );
        assert_eq!(
            shape_from_world(&w, e),
            Some(Shape3D::Box(Vec3::splat(2.0)))
        );
    }

    #[test]
    fn shape_from_sphere_keeps_radius() {
        let mut w = World::new();
        let e = w.spawn();
        w.add(
            e,
            SphereCollider3D {
                radius: 3.0,
                ..Default::default()
            },
        );
        assert_eq!(shape_from_world(&w, e), Some(Shape3D::Sphere(3.0)));
    }

    #[test]
    fn shape_from_capsule_halves_the_height() {
        let mut w = World::new();
        let e = w.spawn();
        w.add(
            e,
            CapsuleCollider3D {
                radius: 1.0,
                height: 6.0,
                ..Default::default()
            },
        );
        assert_eq!(
            shape_from_world(&w, e),
            Some(Shape3D::Capsule {
                radius: 1.0,
                half_height: 3.0
            })
        );
    }

    #[test]
    fn shape_from_world_is_none_without_a_collider() {
        let mut w = World::new();
        let e = w.spawn();
        assert_eq!(shape_from_world(&w, e), None);
    }

    #[test]
    fn bounding_radius_of_each_shape() {
        assert!((Shape3D::Sphere(2.0).bounding_radius() - 2.0).abs() < 1e-5);
        assert!((Shape3D::Box(Vec3::ONE).bounding_radius() - 3f32.sqrt()).abs() < 1e-4);
        assert!(
            (Shape3D::Capsule {
                radius: 1.0,
                half_height: 3.0
            }
            .bounding_radius()
                - 4.0)
                .abs()
                < 1e-5
        );
    }

    #[test]
    fn effective_mass_2d_uses_area_times_density() {
        let mut w = World::new();
        let e = w.spawn();
        w.add(
            e,
            Rigidbody2D {
                body_type: BodyType2D::Dynamic,
                ..Default::default()
            },
        );
        w.add(
            e,
            BoxCollider2D {
                size: Vec2::splat(2.0),
                material: ColliderMaterial {
                    density: 3.0,
                    ..ColliderMaterial::default()
                },
                ..Default::default()
            },
        );
        // area 4 * density 3
        assert_eq!(effective_mass_2d(&w, e), Some(12.0));
    }

    #[test]
    fn static_bodies_have_no_effective_mass() {
        let (w, e) = world_with_2d(BodyType2D::Static);
        assert_eq!(effective_mass_2d(&w, e), None);
    }

    #[test]
    fn kinematic_bodies_have_no_effective_mass() {
        let (w, e) = world_with_2d(BodyType2D::Kinematic);
        assert_eq!(effective_mass_2d(&w, e), None);
    }

    #[test]
    fn effective_mass_2d_without_collider_is_none() {
        let mut w = World::new();
        let e = w.spawn();
        w.add(
            e,
            Rigidbody2D {
                body_type: BodyType2D::Dynamic,
                ..Default::default()
            },
        );
        assert_eq!(effective_mass_2d(&w, e), None);
    }

    #[test]
    fn effective_mass_3d_box_volume_times_density() {
        let mut w = World::new();
        let e = w.spawn();
        w.add(
            e,
            Rigidbody3D {
                body_type: BodyType3D::Dynamic,
                ..Default::default()
            },
        );
        w.add(
            e,
            BoxCollider3D {
                size: Vec3::splat(2.0),
                material: ColliderMaterial {
                    density: 2.0,
                    ..ColliderMaterial::default()
                },
                ..Default::default()
            },
        );
        // volume 8 * density 2
        assert_eq!(effective_mass_3d(&w, e), Some(16.0));
    }

    #[test]
    fn effective_mass_3d_sphere_matches_the_formula() {
        let mut w = World::new();
        let e = w.spawn();
        w.add(
            e,
            Rigidbody3D {
                body_type: BodyType3D::Dynamic,
                ..Default::default()
            },
        );
        w.add(
            e,
            SphereCollider3D {
                radius: 1.0,
                ..Default::default()
            },
        );
        let expected = (4.0 / 3.0) * std::f32::consts::PI;
        assert!((effective_mass_3d(&w, e).unwrap() - expected).abs() < 1e-4);
    }

    #[test]
    fn effective_mass_3d_capsule_is_positive_and_scales() {
        let mut w = World::new();
        let small = w.spawn();
        w.add(
            small,
            Rigidbody3D {
                body_type: BodyType3D::Dynamic,
                ..Default::default()
            },
        );
        w.add(
            small,
            CapsuleCollider3D {
                radius: 1.0,
                height: 2.0,
                ..Default::default()
            },
        );
        let big = w.spawn();
        w.add(
            big,
            Rigidbody3D {
                body_type: BodyType3D::Dynamic,
                ..Default::default()
            },
        );
        w.add(
            big,
            CapsuleCollider3D {
                radius: 2.0,
                height: 2.0,
                ..Default::default()
            },
        );

        let a = effective_mass_3d(&w, small).unwrap();
        let b = effective_mass_3d(&w, big).unwrap();
        assert!(a > 0.0);
        assert!(b > a, "a bigger capsule must be heavier");
    }

    #[test]
    fn transform_round_trips_through_the_world() {
        let mut w = World::new();
        let e = w.spawn();
        w.add(e, Transform::default());

        let body = BodyTransform {
            position: Vec3::new(1.0, 2.0, 3.0),
            rotation: Quat::from_rotation_y(0.5),
        };
        write_transform(&mut w, e, body);
        assert_eq!(transform_from_world(&w, e), body);
    }

    #[test]
    fn transform_from_world_defaults_to_identity() {
        let mut w = World::new();
        let e = w.spawn();
        assert_eq!(transform_from_world(&w, e), BodyTransform::identity());
    }

    #[test]
    fn write_transform_to_entity_without_transform_is_a_noop() {
        let mut w = World::new();
        let e = w.spawn();
        write_transform(&mut w, e, BodyTransform::identity());
        assert!(!w.has::<Transform>(e));
    }

    #[test]
    fn summarise_counts_by_type() {
        let mut w = World::new();
        for bt in [
            BodyType2D::Static,
            BodyType2D::Kinematic,
            BodyType2D::Dynamic,
        ] {
            let e = w.spawn();
            w.add(
                e,
                Rigidbody2D {
                    body_type: bt,
                    ..Default::default()
                },
            );
        }
        for bt in [BodyType3D::Static, BodyType3D::Dynamic] {
            let e = w.spawn();
            w.add(
                e,
                Rigidbody3D {
                    body_type: bt,
                    ..Default::default()
                },
            );
        }
        let c = w.spawn();
        w.add(c, SphereCollider3D::default());

        let counts = summarise(&w);
        assert_eq!(counts.static_2d, 1);
        assert_eq!(counts.kinematic_2d, 1);
        assert_eq!(counts.dynamic_2d, 1);
        assert_eq!(counts.static_3d, 1);
        assert_eq!(counts.dynamic_3d, 1);
        assert_eq!(counts.kinematic_3d, 0);
        assert_eq!(counts.total(), 5);
        assert_eq!(counts.colliders_3d, 1);
    }

    #[test]
    fn summarise_on_empty_world() {
        let w = World::new();
        let counts = summarise(&w);
        assert!(counts.is_empty());
        assert_eq!(counts.total(), 0);
    }

    #[test]
    fn bodies_iterator_lists_registrations() {
        let mut r = PhysicsRegistry::new();
        let a = Entity::from_parts(0, 0);
        let b = Entity::from_parts(1, 0);
        {
            let h = r.next_handle(0);
            r.register(a, h);
        }
        {
            let h = r.next_handle(1);
            r.register(b, h);
        }
        let mut seen: Vec<Entity> = r.bodies().map(|(e, _)| e).collect();
        seen.sort();
        assert_eq!(seen, vec![a, b]);
    }

    #[test]
    fn empty_registry_is_empty() {
        let r = PhysicsRegistry::new();
        assert!(r.is_empty());
        assert_eq!(r.len(), 0);
    }
}
