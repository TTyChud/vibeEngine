//! Prints a small scene as YAML, so the on-disk format is inspectable.
//!
//! Run with: cargo run -p vibe-scene --example dump

use glam::Vec2;
use vibe_ecs::World;
use vibe_ecs::components::*;
use vibe_scene::{world_to_yaml, yaml_to_world};

fn main() {
    let mut world = World::new();

    let player = world.spawn();
    world.add(player, Tag::new("player"));
    world.add(player, Transform::at_2d(64.0, 128.0));
    world.add(
        player,
        SpriteRenderer {
            texture_index: 0,
            size: Vec2::splat(32.0),
            sort_order: 10,
            ..Default::default()
        },
    );
    world.add(
        player,
        Rigidbody2D {
            body_type: BodyType2D::Dynamic,
            mass: 2.0,
            gravity_scale: 1.0,
            ..Default::default()
        },
    );
    world.add(player, BoxCollider2D::new(Vec2::splat(1.0)));

    let camera = world.spawn();
    world.add(camera, Tag::new("main_camera"));
    world.add(camera, Transform::at_2d(320.0, 180.0));
    world.add(camera, Camera::default());

    let light = world.spawn();
    world.add(light, Tag::new("sun"));
    world.add(
        light,
        DirectionalLight {
            intensity: 1.2,
            ..Default::default()
        },
    );

    let yaml = world_to_yaml(&world);
    println!("{yaml}");

    // Prove it round-trips.
    match yaml_to_world(&yaml) {
        Ok(back) => println!(
            "// round trip: {} entities, {} tags, {} sprites",
            back.len(),
            back.count::<Tag>(),
            back.count::<SpriteRenderer>()
        ),
        Err(e) => println!("// round trip failed: {e}"),
    }
}
