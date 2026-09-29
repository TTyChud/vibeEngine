//! 2D rigid body simulation, backed by Rapier.

use std::collections::HashMap;

use glam::{Vec2, Vec3};
// Aliased rather than globbed: rapier2d's prelude re-exports glam type names
// that collide with this file's own glam imports.
use rapier2d::prelude::{
    AngVector, ColliderBuilder, ColliderHandle, PhysicsWorld, Real, RigidBodyBuilder,
    RigidBodyHandle, RigidBodyType, Vector,
};
use vibe_ecs::{Entity, World, components::*};

use crate::error::PhysicsError;
use crate::registry::{BodyTransform, transform_from_world, write_transform};

/// A 2D physics world driven by ECS components.
///
/// Bodies and colliders are created from components, stepped once per frame, and
/// their transforms written back. The handle map is what makes that safe: a
/// handle is only reused after the old body is destroyed, so a stale one is
/// caught rather than silently addressing a different body.
pub struct Physics2D {
    world: PhysicsWorld,
    bodies: HashMap<Entity, RigidBodyHandle>,
    colliders: HashMap<Entity, ColliderHandle>,
}

impl Default for Physics2D {
    fn default() -> Self {
        Physics2D::new()
    }
}

impl Physics2D {
    /// A 2D world with default gravity.
    pub fn new() -> Physics2D {
        Physics2D {
            world: PhysicsWorld::new(),
            bodies: HashMap::new(),
            colliders: HashMap::new(),
        }
    }

    /// A 2D world with a chosen gravity.
    pub fn with_gravity(gravity: Vec2) -> Physics2D {
        let mut sim = Physics2D::new();
        sim.set_gravity(gravity);
        sim
    }

    /// Replace the world's gravity.
    pub fn set_gravity(&mut self, gravity: Vec2) {
        self.world.gravity = vector(gravity.x, gravity.y);
    }

    /// The current gravity.
    pub fn gravity(&self) -> Vec2 {
        let g = self.world.gravity;
        Vec2::new(g.x, g.y)
    }

    /// Set the fixed timestep used by [`Physics2D::step`].
    ///
    /// # Errors
    ///
    /// Rejects a non-positive timestep, which would make the solver divide by
    /// zero.
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

    /// Create or update the body for an entity from its components.
    ///
    /// Returns the body handle. Calling this again for the same entity updates
    /// the existing body rather than creating a second one.
    ///
    /// # Errors
    ///
    /// Returns [`PhysicsError::NoCollider`] when the entity has a body but no
    /// [`BoxCollider2D`], since a body with no shape would fall through the
    /// world.
    pub fn sync_body(
        &mut self,
        world_ecs: &World,
        entity: Entity,
    ) -> Result<RigidBodyHandle, PhysicsError> {
        let rb = world_ecs
            .get::<Rigidbody2D>(entity)
            .ok_or_else(|| PhysicsError::NoBody {
                entity: entity.to_string(),
                kind: "2D",
            })?;
        let collider = world_ecs
            .get::<BoxCollider2D>(entity)
            .ok_or_else(|| PhysicsError::NoCollider(entity.to_string()))?;
        if collider.size.x <= 0.0 || collider.size.y <= 0.0 {
            return Err(PhysicsError::DegenerateShape {
                entity: entity.to_string(),
                shape: "BoxCollider2D",
            });
        }

        let t = transform_from_world(world_ecs, entity);
        let body_type = match rb.body_type {
            BodyType2D::Static => RigidBodyType::Fixed,
            BodyType2D::Kinematic => RigidBodyType::KinematicPositionBased,
            BodyType2D::Dynamic => RigidBodyType::Dynamic,
        };

        if let Some(&existing) = self.bodies.get(&entity) {
            if let Some(mut body) = self.world.remove_body(existing) {
                body.set_body_type(body_type, true);
                body.set_translation(vector(t.position.x, t.position.y), true);
                body.set_linear_damping(rb.linear_damping as Real);
                body.set_angular_damping(rb.angular_damping as Real);
                let handle = self.world.insert_body(body);
                self.bodies.insert(entity, handle);
                return Ok(handle);
            }
            // The handle went stale; fall through and create a fresh body.
            self.bodies.remove(&entity);
            self.colliders.remove(&entity);
        }

        let mut builder = RigidBodyBuilder::new(body_type)
            .translation(vector(t.position.x, t.position.y))
            .gravity_scale(rb.gravity_scale as Real);
        if rb.is_rotatable {
            builder = builder.rotation(rb_ang(t.rotation));
        } else {
            builder = builder.lock_rotations();
        }
        let handle = self.world.insert_body(builder.build());
        self.bodies.insert(entity, handle);

        let shape = ColliderBuilder::cuboid(
            (collider.size.x * 0.5) as Real,
            (collider.size.y * 0.5) as Real,
        );
        let c = shape
            .density(collider.material.density as Real)
            .friction(collider.material.friction as Real)
            .restitution(collider.material.restitution as Real)
            .sensor(rb.is_sensor);
        let collider_handle = self.world.insert_collider(c, Some(handle));
        self.colliders.insert(entity, collider_handle);

        Ok(handle)
    }

    /// Destroy an entity's body and collider.
    pub fn remove_body(&mut self, entity: Entity) {
        if let Some(handle) = self.bodies.remove(&entity) {
            // Removing the body takes its attached colliders with it.
            self.world.remove_body(handle);
        }
        if let Some(handle) = self.colliders.remove(&entity) {
            self.world.remove_collider(handle);
        }
    }

    /// Remove bodies for entities that no longer exist in the ECS.
    ///
    /// Returns how many were dropped.
    pub fn prune(&mut self, world_ecs: &World) -> usize {
        let dead: Vec<Entity> = self
            .bodies
            .keys()
            .copied()
            .filter(|e| !world_ecs.is_alive(*e))
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
    pub fn step_and_sync(&mut self, world_ecs: &mut World) {
        self.step();
        let entries: Vec<(Entity, BodyTransform)> = self
            .bodies
            .iter()
            .filter_map(|(entity, handle)| {
                let body = self.world.bodies.get(*handle)?;
                let t = body.translation();
                let angle = body.rotation().angle();
                Some((
                    *entity,
                    BodyTransform {
                        position: Vec3::new(t.x, t.y, 0.0),
                        rotation: glam::Quat::from_rotation_z(angle as f32),
                    },
                ))
            })
            .collect();
        for (entity, transform) in entries {
            if world_ecs.is_alive(entity) {
                write_transform(world_ecs, entity, transform);
            }
        }
    }

    /// A body's linear velocity.
    pub fn linvel(&self, entity: Entity) -> Option<Vec2> {
        let handle = *self.bodies.get(&entity)?;
        let body = self.world.bodies.get(handle)?;
        let v = body.linvel();
        Some(Vec2::new(v.x, v.y))
    }

    /// Set a body's linear velocity.
    pub fn set_linvel(&mut self, entity: Entity, velocity: Vec2) -> bool {
        let Some(&handle) = self.bodies.get(&entity) else {
            return false;
        };
        if let Some(body) = self.world.bodies.get_mut(handle) {
            body.set_linvel(vector(velocity.x, velocity.y), true);
            true
        } else {
            false
        }
    }

    /// Apply an instantaneous impulse, changing velocity directly.
    pub fn apply_impulse(&mut self, entity: Entity, impulse: Vec2) -> bool {
        let Some(&handle) = self.bodies.get(&entity) else {
            return false;
        };
        if let Some(body) = self.world.bodies.get_mut(handle) {
            body.apply_impulse(vector(impulse.x, impulse.y), true);
            true
        } else {
            false
        }
    }

    /// A body's position.
    pub fn position(&self, entity: Entity) -> Option<Vec2> {
        let handle = *self.bodies.get(&entity)?;
        let t = self.world.bodies.get(handle)?.translation();
        Some(Vec2::new(t.x, t.y))
    }

    /// Teleport a body, waking it so it is simulated again.
    pub fn set_position(&mut self, entity: Entity, position: Vec2) -> bool {
        let Some(&handle) = self.bodies.get(&entity) else {
            return false;
        };
        if let Some(body) = self.world.bodies.get_mut(handle) {
            body.set_translation(vector(position.x, position.y), true);
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

    /// Borrow the underlying Rapier world, for queries the engine does not wrap.
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

fn vector(x: f32, y: f32) -> Vector {
    Vector::new(x as Real, y as Real)
}

fn rb_ang(q: glam::Quat) -> AngVector {
    // A 2D body only has a Z rotation, so take the angle from the quaternion's
    // Z component rather than decomposing a full 3D rotation. In 2D
    // `AngVector` is the scalar itself.
    (2.0 * q.z.atan2(q.w)) as Real
}

#[cfg(test)]
mod tests {
    use super::*;

    fn body_world() -> (World, Entity) {
        let mut w = World::new();
        let e = w.spawn();
        w.add(e, Transform::default());
        w.add(
            e,
            Rigidbody2D {
                body_type: BodyType2D::Dynamic,
                ..Default::default()
            },
        );
        w.add(e, BoxCollider2D::new(Vec2::splat(1.0)));
        (w, e)
    }

    fn static_world() -> (World, Entity) {
        let mut w = World::new();
        let e = w.spawn();
        w.add(e, Transform::default());
        w.add(
            e,
            Rigidbody2D {
                body_type: BodyType2D::Static,
                ..Default::default()
            },
        );
        w.add(e, BoxCollider2D::new(Vec2::splat(1.0)));
        (w, e)
    }

    #[test]
    fn new_world_has_default_gravity() {
        let sim = Physics2D::new();
        assert!(
            sim.gravity().y < 0.0,
            "2D default gravity should point down"
        );
        assert!(sim.is_empty());
    }

    #[test]
    fn gravity_can_be_set() {
        let mut sim = Physics2D::with_gravity(Vec2::new(0.0, 10.0));
        assert_eq!(sim.gravity(), Vec2::new(0.0, 10.0));
        sim.set_gravity(Vec2::new(1.0, 2.0));
        assert_eq!(sim.gravity(), Vec2::new(1.0, 2.0));
    }

    #[test]
    fn timestep_round_trips() {
        let mut sim = Physics2D::new();
        sim.set_timestep(0.02).unwrap();
        assert!((sim.timestep() - 0.02).abs() < 1e-6);
    }

    #[test]
    fn zero_timestep_is_rejected() {
        let mut sim = Physics2D::new();
        assert!(matches!(
            sim.set_timestep(0.0),
            Err(PhysicsError::InvalidTimestep(_))
        ));
        assert!(matches!(
            sim.set_timestep(-1.0),
            Err(PhysicsError::InvalidTimestep(_))
        ));
        assert!(matches!(
            sim.set_timestep(f32::NAN),
            Err(PhysicsError::InvalidTimestep(_))
        ));
    }

    #[test]
    fn sync_body_creates_a_body() {
        let (w, e) = body_world();
        let mut sim = Physics2D::new();
        sim.sync_body(&w, e).unwrap();
        assert!(sim.has_body(e));
        assert_eq!(sim.len(), 1);
    }

    #[test]
    fn body_without_rigidbody_component_is_rejected() {
        let mut w = World::new();
        let e = w.spawn();
        w.add(e, BoxCollider2D::new(Vec2::ONE));
        let mut sim = Physics2D::new();
        assert!(matches!(
            sim.sync_body(&w, e),
            Err(PhysicsError::NoBody { .. })
        ));
    }

    #[test]
    fn body_without_collider_is_rejected() {
        let mut w = World::new();
        let e = w.spawn();
        w.add(e, Rigidbody2D::default());
        let mut sim = Physics2D::new();
        assert!(matches!(
            sim.sync_body(&w, e),
            Err(PhysicsError::NoCollider(_))
        ));
    }

    #[test]
    fn zero_size_collider_is_rejected() {
        let mut w = World::new();
        let e = w.spawn();
        w.add(e, Rigidbody2D::default());
        w.add(e, BoxCollider2D::new(Vec2::ZERO));
        let mut sim = Physics2D::new();
        assert!(matches!(
            sim.sync_body(&w, e),
            Err(PhysicsError::DegenerateShape { .. })
        ));
    }

    #[test]
    fn dynamic_body_falls_under_gravity() {
        let (w, e) = body_world();
        let mut sim = Physics2D::new();
        sim.sync_body(&w, e).unwrap();
        let before = sim.position(e).unwrap();
        for _ in 0..30 {
            sim.step();
        }
        let after = sim.position(e).unwrap();
        assert!(
            after.y < before.y,
            "a dynamic body should fall: {before:?} -> {after:?}"
        );
    }

    #[test]
    fn static_body_does_not_move() {
        let (w, e) = static_world();
        let mut sim = Physics2D::new();
        sim.sync_body(&w, e).unwrap();
        let before = sim.position(e).unwrap();
        for _ in 0..30 {
            sim.step();
        }
        assert!(
            (sim.position(e).unwrap().y - before.y).abs() < 1e-5,
            "a fixed body must not move"
        );
    }

    #[test]
    fn step_and_sync_writes_back_to_the_ecs() {
        let (mut w, e) = body_world();
        let mut sim = Physics2D::new();
        sim.sync_body(&w, e).unwrap();
        for _ in 0..10 {
            sim.step_and_sync(&mut w);
        }
        let t = w.get::<Transform>(e).unwrap();
        assert!(
            t.y < 0.0,
            "the ecs transform should show the fall, got {}",
            t.y
        );
    }

    #[test]
    fn a_falling_body_lands_on_a_static_floor() {
        let (mut w, ball) = body_world();
        let floor = w.spawn();
        w.add(floor, Transform::default());
        w.get_mut::<Transform>(floor).unwrap().y = -5.0;
        w.add(
            floor,
            Rigidbody2D {
                body_type: BodyType2D::Static,
                ..Default::default()
            },
        );
        w.add(floor, BoxCollider2D::new(Vec2::splat(10.0)));

        let mut sim = Physics2D::new();
        sim.sync_body(&w, ball).unwrap();
        sim.sync_body(&w, floor).unwrap();
        for _ in 0..200 {
            sim.step_and_sync(&mut w);
        }
        let y = w.get::<Transform>(ball).unwrap().y;
        // The floor's top surface is at -4.5; the ball's centre rests on it.
        assert!(
            y > -5.0,
            "the ball should rest on the floor, not sink through: y={y}"
        );
    }

    #[test]
    fn sync_body_twice_does_not_duplicate() {
        let (w, e) = body_world();
        let mut sim = Physics2D::new();
        sim.sync_body(&w, e).unwrap();
        sim.sync_body(&w, e).unwrap();
        assert_eq!(sim.len(), 1, "re-syncing must update, not add");
    }

    #[test]
    fn remove_body_drops_it() {
        let (w, e) = body_world();
        let mut sim = Physics2D::new();
        sim.sync_body(&w, e).unwrap();
        sim.remove_body(e);
        assert!(!sim.has_body(e));
        assert!(sim.is_empty());
    }

    #[test]
    fn prune_removes_bodies_for_dead_entities() {
        let (w, e) = body_world();
        let mut sim = Physics2D::new();
        sim.sync_body(&w, e).unwrap();
        let mut w = w;
        w.despawn(e);
        assert_eq!(sim.prune(&w), 1);
        assert!(sim.is_empty());
    }

    #[test]
    fn set_position_teleports() {
        let (w, e) = body_world();
        let mut sim = Physics2D::new();
        sim.sync_body(&w, e).unwrap();
        assert!(sim.set_position(e, Vec2::new(3.0, 4.0)));
        let p = sim.position(e).unwrap();
        assert!((p.x - 3.0).abs() < 1e-5 && (p.y - 4.0).abs() < 1e-5);
    }

    #[test]
    fn set_linvel_changes_velocity() {
        let (w, e) = body_world();
        let mut sim = Physics2D::new();
        sim.sync_body(&w, e).unwrap();
        assert!(sim.set_linvel(e, Vec2::new(5.0, 0.0)));
        let v = sim.linvel(e).unwrap();
        assert!((v.x - 5.0).abs() < 1e-4, "got {v:?}");
    }

    #[test]
    fn impulse_accelerates_a_body() {
        let (w, e) = body_world();
        let mut sim = Physics2D::new();
        sim.sync_body(&w, e).unwrap();
        sim.set_linvel(e, Vec2::ZERO);
        sim.apply_impulse(e, Vec2::new(10.0, 0.0));
        assert!(
            sim.linvel(e).unwrap().x > 0.0,
            "an impulse should add velocity"
        );
    }

    #[test]
    fn operations_on_unknown_entities_return_false() {
        let mut sim = Physics2D::new();
        let ghost = Entity::from_parts(99, 0);
        assert!(!sim.has_body(ghost));
        assert!(!sim.set_position(ghost, Vec2::ZERO));
        assert!(!sim.set_linvel(ghost, Vec2::ZERO));
        assert!(!sim.apply_impulse(ghost, Vec2::ZERO));
        assert_eq!(sim.position(ghost), None);
        assert_eq!(sim.linvel(ghost), None);
    }

    #[test]
    fn entities_iterator_lists_bodies() {
        let (mut w, a) = body_world();
        let b = w.spawn();
        w.add(b, Transform::default());
        w.add(b, Rigidbody2D::default());
        w.add(b, BoxCollider2D::new(Vec2::ONE));

        let mut sim = Physics2D::new();
        sim.sync_body(&w, a).unwrap();
        sim.sync_body(&w, b).unwrap();
        let mut seen: Vec<Entity> = sim.entities().collect();
        seen.sort();
        assert_eq!(seen, vec![a, b]);
    }

    #[test]
    fn destroy_empties_the_world() {
        let (w, e) = body_world();
        let mut sim = Physics2D::new();
        sim.sync_body(&w, e).unwrap();
        sim.destroy();
        assert!(sim.is_empty());
        assert!(!sim.has_body(e));
    }

    #[test]
    fn kinematic_body_moves_only_when_told() {
        let mut w = World::new();
        let e = w.spawn();
        w.add(e, Transform::default());
        w.add(
            e,
            Rigidbody2D {
                body_type: BodyType2D::Kinematic,
                ..Default::default()
            },
        );
        w.add(e, BoxCollider2D::new(Vec2::ONE));

        let mut sim = Physics2D::new();
        sim.sync_body(&w, e).unwrap();
        let before = sim.position(e).unwrap();
        for _ in 0..30 {
            sim.step();
        }
        let after = sim.position(e).unwrap();
        assert!(
            (after.x - before.x).abs() < 1e-4,
            "kinematic bodies ignore gravity"
        );
    }

    #[test]
    fn a_rotatable_body_is_created_with_its_rotation() {
        let mut w = World::new();
        let e = w.spawn();
        w.add(e, Transform::default());
        w.add(
            e,
            Rigidbody2D {
                body_type: BodyType2D::Dynamic,
                is_rotatable: true,
                ..Default::default()
            },
        );
        w.add(e, BoxCollider2D::new(Vec2::ONE));
        // Set the rotation before the first sync, so the builder receives it.
        w.get_mut::<Transform>(e).unwrap().rotation = glam::Quat::from_rotation_z(0.5);

        let mut sim = Physics2D::new();
        sim.sync_body(&w, e).unwrap();
        for _ in 0..10 {
            sim.step();
        }
        // Gravity is still -Y, so a rotated body must still fall.
        assert!(sim.position(e).unwrap().y < 0.0);
    }

    #[test]
    fn raw_world_is_accessible() {
        let (w, e) = body_world();
        let mut sim = Physics2D::new();
        sim.sync_body(&w, e).unwrap();
        assert_eq!(sim.raw().bodies.len(), 1);
        sim.raw_mut().step();
    }

    #[test]
    fn many_bodies_simulate_together() {
        let mut w = World::new();
        let mut made = Vec::new();
        for i in 0..20 {
            let e = w.spawn();
            w.add(e, Transform::at_2d(i as f32, 10.0));
            w.add(
                e,
                Rigidbody2D {
                    body_type: BodyType2D::Dynamic,
                    ..Default::default()
                },
            );
            w.add(e, BoxCollider2D::new(Vec2::ONE));
            made.push(e);
        }
        let mut sim = Physics2D::new();
        for e in &made {
            sim.sync_body(&w, *e).unwrap();
        }
        assert_eq!(sim.len(), 20);
        for _ in 0..30 {
            sim.step();
        }
        for e in &made {
            assert!(
                sim.position(*e).unwrap().y < 10.0,
                "every body should have fallen"
            );
        }
    }
}
