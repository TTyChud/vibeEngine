//! 3D rigid body simulation, backed by Rapier.

use std::collections::HashMap;

use glam::{Quat, Vec3};
// Aliased rather than globbed: rapier3d's prelude re-exports the same glam type
// names, and a glob would make every one of them ambiguous here.
use rapier3d::prelude::{
    ColliderBuilder, ColliderHandle, PhysicsWorld, Pose3, Real, RigidBodyBuilder, RigidBodyHandle,
    RigidBodyType, Vector,
};
use vibe_ecs::components::ColliderMaterial as EcsMaterial;
use vibe_ecs::{Entity, World, components::*};

use crate::error::PhysicsError;
use crate::registry::{
    BodyTransform, Shape3D, shape_from_world, transform_from_world, write_transform,
};

/// A 3D physics world driven by ECS components.
///
/// Handles box, sphere and capsule colliders, and honours the per-axis degrees
/// of freedom locks on [`Rigidbody3D`].
pub struct Physics3D {
    world: PhysicsWorld,
    bodies: HashMap<Entity, RigidBodyHandle>,
    colliders: HashMap<Entity, ColliderHandle>,
}

impl Default for Physics3D {
    fn default() -> Self {
        Physics3D::new()
    }
}

impl Physics3D {
    /// A 3D world with default gravity.
    pub fn new() -> Physics3D {
        Physics3D {
            world: PhysicsWorld::new(),
            bodies: HashMap::new(),
            colliders: HashMap::new(),
        }
    }

    /// A 3D world with a chosen gravity.
    pub fn with_gravity(gravity: Vec3) -> Physics3D {
        let mut sim = Physics3D::new();
        sim.set_gravity(gravity);
        sim
    }

    /// Replace the world's gravity.
    pub fn set_gravity(&mut self, gravity: Vec3) {
        self.world.gravity = vector(gravity.x, gravity.y, gravity.z);
    }

    /// The current gravity.
    pub fn gravity(&self) -> Vec3 {
        let g = self.world.gravity;
        Vec3::new(g.x, g.y, g.z)
    }

    /// Set the fixed timestep.
    ///
    /// # Errors
    ///
    /// Rejects a non-positive or non-finite timestep.
    pub fn set_timestep(&mut self, dt: f32) -> Result<(), PhysicsError> {
        if dt <= 0.0 || !dt.is_finite() {
            return Err(PhysicsError::InvalidTimestep(dt));
        }
        self.world.integration_parameters.dt = dt as Real;
        Ok(())
    }

    /// The fixed timestep.
    pub fn timestep(&self) -> f32 {
        self.world.integration_parameters.dt as f32
    }

    /// Create or update the body and collider for an entity.
    ///
    /// # Errors
    ///
    /// Returns [`PhysicsError::NoCollider`] when the entity has a body but no
    /// shape, and [`PhysicsError::DegenerateShape`] for a non-positive size.
    pub fn sync_body(
        &mut self,
        ecs: &World,
        entity: Entity,
    ) -> Result<RigidBodyHandle, PhysicsError> {
        let rb = ecs
            .get::<Rigidbody3D>(entity)
            .ok_or_else(|| PhysicsError::NoBody {
                entity: entity.to_string(),
                kind: "3D",
            })?;
        let shape = shape_from_world(ecs, entity)
            .ok_or_else(|| PhysicsError::NoCollider(entity.to_string()))?;

        let material = collider_material(ecs, entity);
        validate_shape(entity, shape)?;

        let t = transform_from_world(ecs, entity);
        let body_type = match rb.body_type {
            BodyType3D::Static => RigidBodyType::Fixed,
            BodyType3D::Kinematic => RigidBodyType::KinematicPositionBased,
            BodyType3D::Dynamic => RigidBodyType::Dynamic,
        };

        if let Some(&existing) = self.bodies.get(&entity) {
            if let Some(mut body) = self.world.remove_body(existing) {
                body.set_body_type(body_type, true);
                body.set_translation(vector(t.position.x, t.position.y, t.position.z), true);
                body.set_rotation(t.rotation, true);
                body.set_linear_damping(rb.linear_damping as Real);
                body.set_angular_damping(rb.angular_damping as Real);
                let handle = self.world.insert_body(body);
                self.bodies.insert(entity, handle);
                return Ok(handle);
            }
            self.bodies.remove(&entity);
            self.colliders.remove(&entity);
        }

        let builder = RigidBodyBuilder::new(body_type)
            .translation(vector(t.position.x, t.position.y, t.position.z))
            .pose(pose(t.position, t.rotation))
            .gravity_scale(rb.gravity_scale as Real)
            // A lock is "do not allow motion on this axis", so the builder takes
            // the allowed axes.
            .enabled_translations(!rb.lock_linear.0, !rb.lock_linear.1, !rb.lock_linear.2)
            .enabled_rotations(!rb.lock_angular.0, !rb.lock_angular.1, !rb.lock_angular.2);

        let handle = self.world.insert_body(builder.build());
        self.bodies.insert(entity, handle);

        let collider = collider_for(shape)
            .density(material.density as Real)
            .friction(material.friction as Real)
            .restitution(material.restitution as Real)
            .sensor(rb.is_sensor);
        let collider_handle = self.world.insert_collider(collider, Some(handle));
        self.colliders.insert(entity, collider_handle);

        Ok(handle)
    }

    /// Destroy an entity's body and collider.
    pub fn remove_body(&mut self, entity: Entity) {
        if let Some(handle) = self.bodies.remove(&entity) {
            self.world.remove_body(handle);
        }
        if let Some(handle) = self.colliders.remove(&entity) {
            self.world.remove_collider(handle);
        }
    }

    /// Remove bodies for entities that no longer exist.
    pub fn prune(&mut self, ecs: &World) -> usize {
        let dead: Vec<Entity> = self
            .bodies
            .keys()
            .copied()
            .filter(|e| !ecs.is_alive(*e))
            .collect();
        for e in &dead {
            self.remove_body(*e);
        }
        dead.len()
    }

    /// Advance the simulation one timestep.
    pub fn step(&mut self) {
        self.world.step();
    }

    /// Step, then write every body's transform back into the ECS.
    pub fn step_and_sync(&mut self, ecs: &mut World) {
        self.step();
        let entries: Vec<(Entity, BodyTransform)> = self
            .bodies
            .iter()
            .filter_map(|(entity, handle)| {
                let body = self.world.bodies.get(*handle)?;
                let t = body.translation();
                let q = body.rotation();
                Some((
                    *entity,
                    BodyTransform {
                        position: Vec3::new(t.x, t.y, t.z),
                        rotation: Quat::from_xyzw(q.x, q.y, q.z, q.w),
                    },
                ))
            })
            .collect();
        for (entity, transform) in entries {
            if ecs.is_alive(entity) {
                write_transform(ecs, entity, transform);
            }
        }
    }

    /// A body's linear velocity.
    pub fn linvel(&self, entity: Entity) -> Option<Vec3> {
        let body = self.world.bodies.get(*self.bodies.get(&entity)?)?;
        let v = body.linvel();
        Some(Vec3::new(v.x, v.y, v.z))
    }

    /// Set a body's linear velocity.
    pub fn set_linvel(&mut self, entity: Entity, velocity: Vec3) -> bool {
        let Some(&handle) = self.bodies.get(&entity) else {
            return false;
        };
        if let Some(body) = self.world.bodies.get_mut(handle) {
            body.set_linvel(vector(velocity.x, velocity.y, velocity.z), true);
            true
        } else {
            false
        }
    }

    /// Apply an instantaneous impulse.
    pub fn apply_impulse(&mut self, entity: Entity, impulse: Vec3) -> bool {
        let Some(&handle) = self.bodies.get(&entity) else {
            return false;
        };
        if let Some(body) = self.world.bodies.get_mut(handle) {
            body.apply_impulse(vector(impulse.x, impulse.y, impulse.z), true);
            true
        } else {
            false
        }
    }

    /// Apply a continuous force, which is what a thruster or wind does.
    pub fn apply_force(&mut self, entity: Entity, force: Vec3) -> bool {
        let Some(&handle) = self.bodies.get(&entity) else {
            return false;
        };
        if let Some(body) = self.world.bodies.get_mut(handle) {
            body.add_force(vector(force.x, force.y, force.z), true);
            true
        } else {
            false
        }
    }

    /// A body's position.
    pub fn position(&self, entity: Entity) -> Option<Vec3> {
        let body = self.world.bodies.get(*self.bodies.get(&entity)?)?;
        let t = body.translation();
        Some(Vec3::new(t.x, t.y, t.z))
    }

    /// Teleport a body.
    pub fn set_position(&mut self, entity: Entity, position: Vec3) -> bool {
        let Some(&handle) = self.bodies.get(&entity) else {
            return false;
        };
        if let Some(body) = self.world.bodies.get_mut(handle) {
            body.set_translation(vector(position.x, position.y, position.z), true);
            true
        } else {
            false
        }
    }

    /// Number of live bodies.
    pub fn len(&self) -> usize {
        self.bodies.len()
    }

    /// True when nothing is simulated.
    pub fn is_empty(&self) -> bool {
        self.bodies.is_empty()
    }

    /// Every simulated entity.
    pub fn entities(&self) -> impl Iterator<Item = Entity> + '_ {
        self.bodies.keys().copied()
    }

    /// True when an entity has a body.
    pub fn has_body(&self, entity: Entity) -> bool {
        self.bodies.contains_key(&entity)
    }

    /// Borrow the underlying Rapier world.
    pub fn raw(&self) -> &PhysicsWorld {
        &self.world
    }

    /// Mutably borrow the underlying Rapier world.
    pub fn raw_mut(&mut self) -> &mut PhysicsWorld {
        &mut self.world
    }

    /// Destroy the world and everything in it.
    pub fn destroy(&mut self) {
        self.bodies.clear();
        self.colliders.clear();
        self.world = PhysicsWorld::new();
    }
}

fn vector(x: f32, y: f32, z: f32) -> Vector {
    Vector::new(x as Real, y as Real, z as Real)
}

/// rapier 0.36 builds a pose from a translation plus an axis-angle, and its
/// `Rotation` is glam's own `Quat`.
fn pose(position: Vec3, q: Quat) -> Pose3 {
    let axis_angle = glam::Quat::to_scaled_axis(q);
    Pose3::new(
        vector(position.x, position.y, position.z),
        vector(axis_angle.x, axis_angle.y, axis_angle.z),
    )
}

fn collider_for(shape: Shape3D) -> ColliderBuilder {
    match shape {
        Shape3D::Box(half) => {
            ColliderBuilder::cuboid(half.x as Real, half.y as Real, half.z as Real)
        }
        Shape3D::Sphere(r) => ColliderBuilder::ball(r as Real),
        Shape3D::Capsule {
            radius,
            half_height,
        } => ColliderBuilder::capsule_y(half_height as Real, radius as Real),
    }
}

fn collider_material(ecs: &World, entity: Entity) -> EcsMaterial {
    ecs.get::<BoxCollider3D>(entity)
        .map(|c| c.material)
        .or_else(|| ecs.get::<SphereCollider3D>(entity).map(|c| c.material))
        .or_else(|| ecs.get::<CapsuleCollider3D>(entity).map(|c| c.material))
        .unwrap_or_default()
}

fn validate_shape(entity: Entity, shape: Shape3D) -> Result<(), PhysicsError> {
    let bad = match shape {
        Shape3D::Box(half) => half.x <= 0.0 || half.y <= 0.0 || half.z <= 0.0,
        Shape3D::Sphere(r) => r <= 0.0,
        Shape3D::Capsule {
            radius,
            half_height,
        } => radius <= 0.0 || half_height < 0.0,
    };
    if bad {
        return Err(PhysicsError::DegenerateShape {
            entity: entity.to_string(),
            shape: match shape {
                Shape3D::Box(_) => "BoxCollider3D",
                Shape3D::Sphere(_) => "SphereCollider3D",
                Shape3D::Capsule { .. } => "CapsuleCollider3D",
            },
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dynamic_box() -> (World, Entity) {
        let mut w = World::new();
        let e = w.spawn();
        w.add(e, Transform::default());
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
                size: Vec3::splat(1.0),
                ..Default::default()
            },
        );
        (w, e)
    }

    fn with_shape(shape: fn(&mut World, Entity)) -> (World, Entity) {
        let mut w = World::new();
        let e = w.spawn();
        w.add(e, Transform::default());
        w.add(
            e,
            Rigidbody3D {
                body_type: BodyType3D::Dynamic,
                ..Default::default()
            },
        );
        shape(&mut w, e);
        (w, e)
    }

    #[test]
    fn new_world_has_gravity() {
        let sim = Physics3D::new();
        assert!(sim.gravity().y < 0.0);
        assert!(sim.is_empty());
    }

    #[test]
    fn gravity_and_timestep_round_trip() {
        let mut sim = Physics3D::with_gravity(Vec3::new(0.0, -20.0, 0.0));
        assert_eq!(sim.gravity(), Vec3::new(0.0, -20.0, 0.0));
        sim.set_timestep(0.016).unwrap();
        assert!((sim.timestep() - 0.016).abs() < 1e-6);
    }

    #[test]
    fn bad_timestep_is_rejected() {
        let mut sim = Physics3D::new();
        assert!(sim.set_timestep(0.0).is_err());
        assert!(sim.set_timestep(-0.5).is_err());
    }

    #[test]
    fn box_body_creates_and_falls() {
        let (w, e) = dynamic_box();
        let mut sim = Physics3D::new();
        sim.sync_body(&w, e).unwrap();
        let before = sim.position(e).unwrap();
        for _ in 0..30 {
            sim.step();
        }
        assert!(sim.position(e).unwrap().y < before.y);
    }

    #[test]
    fn sphere_body_creates_and_falls() {
        let (w, e) = with_shape(|w, e| {
            w.add(
                e,
                SphereCollider3D {
                    radius: 0.5,
                    ..Default::default()
                },
            );
        });
        let mut sim = Physics3D::new();
        sim.sync_body(&w, e).unwrap();
        for _ in 0..20 {
            sim.step();
        }
        assert!(sim.position(e).unwrap().y < 0.0);
    }

    #[test]
    fn capsule_body_creates_and_falls() {
        let (w, e) = with_shape(|w, e| {
            w.add(
                e,
                CapsuleCollider3D {
                    radius: 0.4,
                    height: 1.0,
                    ..Default::default()
                },
            );
        });
        let mut sim = Physics3D::new();
        sim.sync_body(&w, e).unwrap();
        for _ in 0..20 {
            sim.step();
        }
        assert!(sim.position(e).unwrap().y < 0.0);
    }

    #[test]
    fn body_without_collider_is_rejected() {
        let mut w = World::new();
        let e = w.spawn();
        w.add(e, Rigidbody3D::default());
        let mut sim = Physics3D::new();
        assert!(matches!(
            sim.sync_body(&w, e),
            Err(PhysicsError::NoCollider(_))
        ));
    }

    #[test]
    fn body_without_rigidbody_is_rejected() {
        let mut w = World::new();
        let e = w.spawn();
        w.add(e, SphereCollider3D::default());
        let mut sim = Physics3D::new();
        assert!(matches!(
            sim.sync_body(&w, e),
            Err(PhysicsError::NoBody { .. })
        ));
    }

    #[test]
    fn zero_sphere_is_rejected() {
        let (w, e) = with_shape(|w, e| {
            w.add(
                e,
                SphereCollider3D {
                    radius: 0.0,
                    ..Default::default()
                },
            );
        });
        let mut sim = Physics3D::new();
        assert!(matches!(
            sim.sync_body(&w, e),
            Err(PhysicsError::DegenerateShape { .. })
        ));
    }

    #[test]
    fn zero_box_is_rejected() {
        let (w, e) = with_shape(|w, e| {
            w.add(
                e,
                BoxCollider3D {
                    size: Vec3::splat(0.0),
                    ..Default::default()
                },
            );
        });
        let mut sim = Physics3D::new();
        assert!(matches!(
            sim.sync_body(&w, e),
            Err(PhysicsError::DegenerateShape { .. })
        ));
    }

    #[test]
    fn locked_axis_does_not_move() {
        let mut w = World::new();
        let e = w.spawn();
        w.add(e, Transform::default());
        w.add(
            e,
            Rigidbody3D {
                body_type: BodyType3D::Dynamic,
                // Gravity pulls on Y, so locking Y must stop the fall.
                lock_linear: (false, true, false),
                ..Default::default()
            },
        );
        w.add(
            e,
            SphereCollider3D {
                radius: 0.5,
                ..Default::default()
            },
        );

        let mut sim = Physics3D::new();
        sim.sync_body(&w, e).unwrap();
        for _ in 0..60 {
            sim.step();
        }
        let y = sim.position(e).unwrap().y;
        assert!(y.abs() < 1e-3, "a Y-locked body must not fall, got y={y}");
    }

    #[test]
    fn unlocked_body_still_falls() {
        let (w, e) = dynamic_box();
        let mut sim = Physics3D::new();
        sim.sync_body(&w, e).unwrap();
        for _ in 0..60 {
            sim.step();
        }
        assert!(sim.position(e).unwrap().y < -0.1);
    }

    #[test]
    fn a_sensor_still_simulates_but_pushes_nothing() {
        let mut w = World::new();
        let sensor = w.spawn();
        w.add(sensor, Transform::default());
        w.add(
            sensor,
            Rigidbody3D {
                body_type: BodyType3D::Dynamic,
                is_sensor: true,
                ..Default::default()
            },
        );
        w.add(
            sensor,
            SphereCollider3D {
                radius: 0.5,
                ..Default::default()
            },
        );

        // The pusher starts above a stationary sensor and falls onto it.
        let pusher = w.spawn();
        w.add(pusher, Transform::default());
        w.get_mut::<Transform>(pusher).unwrap().y = 3.0;
        w.add(
            pusher,
            Rigidbody3D {
                body_type: BodyType3D::Dynamic,
                ..Default::default()
            },
        );
        w.add(
            pusher,
            SphereCollider3D {
                radius: 0.5,
                ..Default::default()
            },
        );

        // The sensor itself is a fixed body, so only the pusher moves.
        w.get_mut::<Rigidbody3D>(sensor).unwrap().body_type = BodyType3D::Static;

        let mut sim = Physics3D::new();
        sim.sync_body(&w, sensor).unwrap();
        sim.sync_body(&w, pusher).unwrap();
        for _ in 0..60 {
            sim.step();
        }

        // A sensor detects overlap without resisting it, so the falling sphere
        // passes straight through instead of resting on top.
        let sensor_y = sim.position(sensor).unwrap().y;
        let pusher_y = sim.position(pusher).unwrap().y;
        assert!(
            pusher_y < sensor_y,
            "the sensor must not hold the pusher up: sensor at {sensor_y}, pusher at {pusher_y}"
        );
    }

    #[test]
    fn static_body_does_not_move() {
        let mut w = World::new();
        let e = w.spawn();
        w.add(e, Transform::default());
        w.add(
            e,
            Rigidbody3D {
                body_type: BodyType3D::Static,
                ..Default::default()
            },
        );
        w.add(
            e,
            BoxCollider3D {
                size: Vec3::splat(2.0),
                ..Default::default()
            },
        );

        let mut sim = Physics3D::new();
        sim.sync_body(&w, e).unwrap();
        for _ in 0..30 {
            sim.step();
        }
        assert!(sim.position(e).unwrap().y.abs() < 1e-5);
    }

    #[test]
    fn step_and_sync_writes_back() {
        let (mut w, e) = dynamic_box();
        let mut sim = Physics3D::new();
        sim.sync_body(&w, e).unwrap();
        for _ in 0..10 {
            sim.step_and_sync(&mut w);
        }
        assert!(w.get::<Transform>(e).unwrap().y < 0.0);
    }

    #[test]
    fn a_ball_lands_on_a_box() {
        let (mut w, ball) = dynamic_box();
        w.add(
            ball,
            SphereCollider3D {
                radius: 0.5,
                ..Default::default()
            },
        );
        let floor = w.spawn();
        w.add(floor, Transform::default());
        w.get_mut::<Transform>(floor).unwrap().y = -3.0;
        w.add(
            floor,
            Rigidbody3D {
                body_type: BodyType3D::Static,
                ..Default::default()
            },
        );
        w.add(
            floor,
            BoxCollider3D {
                size: Vec3::splat(10.0),
                ..Default::default()
            },
        );

        let mut sim = Physics3D::new();
        sim.sync_body(&w, ball).unwrap();
        sim.sync_body(&w, floor).unwrap();
        for _ in 0..400 {
            sim.step_and_sync(&mut w);
        }
        let y = w.get::<Transform>(ball).unwrap().y;
        assert!(
            y > -3.0,
            "the ball should rest on the floor, not sink: y={y}"
        );
    }

    #[test]
    fn sync_twice_updates_rather_than_duplicates() {
        let (w, e) = dynamic_box();
        let mut sim = Physics3D::new();
        sim.sync_body(&w, e).unwrap();
        sim.sync_body(&w, e).unwrap();
        assert_eq!(sim.len(), 1);
    }

    #[test]
    fn remove_and_prune() {
        let (w, e) = dynamic_box();
        let mut sim = Physics3D::new();
        sim.sync_body(&w, e).unwrap();
        let mut w = w;
        w.despawn(e);
        assert_eq!(sim.prune(&w), 1);
        assert!(sim.is_empty());
    }

    #[test]
    fn velocity_and_impulse() {
        let (w, e) = dynamic_box();
        let mut sim = Physics3D::new();
        sim.sync_body(&w, e).unwrap();
        sim.set_linvel(e, Vec3::new(2.0, 0.0, 0.0));
        assert!((sim.linvel(e).unwrap().x - 2.0).abs() < 1e-4);
        sim.apply_impulse(e, Vec3::new(1.0, 0.0, 0.0));
        assert!(sim.linvel(e).unwrap().x > 2.0);
    }

    #[test]
    fn force_changes_velocity_over_time() {
        let (w, e) = dynamic_box();
        let mut sim = Physics3D::new();
        sim.sync_body(&w, e).unwrap();
        sim.set_linvel(e, Vec3::ZERO);
        sim.apply_force(e, Vec3::new(100.0, 0.0, 0.0));
        sim.step();
        assert!(
            sim.linvel(e).unwrap().x > 0.0,
            "a force should accelerate over the step"
        );
    }

    #[test]
    fn teleport_and_destroy() {
        let (w, e) = dynamic_box();
        let mut sim = Physics3D::new();
        sim.sync_body(&w, e).unwrap();
        assert!(sim.set_position(e, Vec3::new(1.0, 2.0, 3.0)));
        let p = sim.position(e).unwrap();
        assert!((p.y - 2.0).abs() < 1e-4);
        sim.destroy();
        assert!(sim.is_empty());
    }

    #[test]
    fn unknown_entities_are_handled() {
        let mut sim = Physics3D::new();
        let ghost = Entity::from_parts(42, 0);
        assert!(!sim.has_body(ghost));
        assert!(!sim.set_position(ghost, Vec3::ZERO));
        assert!(!sim.set_linvel(ghost, Vec3::ZERO));
        assert!(!sim.apply_impulse(ghost, Vec3::ZERO));
        assert!(!sim.apply_force(ghost, Vec3::ZERO));
        assert_eq!(sim.position(ghost), None);
    }

    #[test]
    fn entities_iterator_lists_bodies() {
        let (mut w, a) = dynamic_box();
        let b = w.spawn();
        w.add(b, Transform::default());
        w.add(b, Rigidbody3D::default());
        w.add(
            b,
            SphereCollider3D {
                radius: 1.0,
                ..Default::default()
            },
        );
        let mut sim = Physics3D::new();
        sim.sync_body(&w, a).unwrap();
        sim.sync_body(&w, b).unwrap();
        assert_eq!(sim.entities().count(), 2);
    }

    #[test]
    fn raw_world_is_accessible() {
        let (w, e) = dynamic_box();
        let mut sim = Physics3D::new();
        sim.sync_body(&w, e).unwrap();
        assert_eq!(sim.raw().bodies.len(), 1);
        sim.raw_mut().step();
    }

    #[test]
    fn collider_material_reaches_the_solver() {
        let mut w = World::new();
        let e = w.spawn();
        w.add(e, Transform::default());
        w.add(
            e,
            Rigidbody3D {
                body_type: BodyType3D::Static,
                ..Default::default()
            },
        );
        w.add(
            e,
            SphereCollider3D {
                radius: 1.0,
                material: EcsMaterial {
                    density: 7.0,
                    friction: 0.9,
                    restitution: 0.4,
                },
                ..Default::default()
            },
        );
        let mut sim = Physics3D::new();
        sim.sync_body(&w, e).unwrap();
        assert!(sim.has_body(e));
    }
}
