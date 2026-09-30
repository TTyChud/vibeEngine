//! The loader against a real glTF document.
//!
//! The unit tests in the library cover the arithmetic, which is where the bugs
//! were. These cover the part that only a real file exercises: the accessor,
//! bufferView and chunk plumbing, where a mistake shows up as a silently
//! missing mesh or a channel that quietly stops animating.

use std::path::Path;

use vibe_gltf::{ChannelValue, LoopMode, Playback};

/// A GLB built to be minimal but complete: one skinned triangle, one material
/// with a base colour, one joint, and one translation channel that lifts the
/// bone from y=0 to y=2 over one second.
fn fixture() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/skinned.glb")
}

fn load() -> vibe_gltf::Model {
    vibe_gltf::load(&fixture()).expect("the fixture must load")
}

#[test]
fn a_real_document_loads_its_mesh() {
    let m = load();
    assert_eq!(m.meshes.len(), 1);
    assert_eq!(m.vertex_count(), 3);
    assert_eq!(m.triangle_count(), 1);
}

#[test]
fn positions_come_through_in_order() {
    let m = load();
    let v = &m.meshes[0].vertices;
    assert_eq!(v[0].position, [0.0, 0.0, 0.0]);
    assert_eq!(v[1].position, [1.0, 0.0, 0.0]);
    assert_eq!(v[2].position, [0.0, 1.0, 0.0]);
}

#[test]
fn normals_come_through() {
    let m = load();
    assert!(
        m.meshes[0]
            .vertices
            .iter()
            .all(|v| v.normal == [0.0, 0.0, 1.0]),
        "{:?}",
        m.meshes[0].vertices[0].normal
    );
}

#[test]
fn joint_indices_and_weights_come_through() {
    let m = load();
    let v = &m.meshes[0].vertices;
    assert_eq!(v[0].joints, [0, 0, 0, 0]);
    assert_eq!(v[1].joints, [1, 0, 0, 0], "vertex 1 is bound to the joint");
    assert_eq!(v[2].joints, [1, 0, 0, 0]);
    assert_eq!(v[1].weights, [1.0, 0.0, 0.0, 0.0]);
}

#[test]
fn indices_come_through() {
    let m = load();
    assert_eq!(m.meshes[0].indices, vec![0, 1, 2]);
}

#[test]
fn the_material_base_colour_comes_through() {
    let m = load();
    assert_eq!(m.materials.len(), 1);
    let mat = m.materials[0];
    assert_eq!(mat.base_color.x, 1.0);
    assert_eq!(mat.base_color.y, 0.0);
    assert_eq!(mat.metallic, 0.0);
    assert_eq!(mat.roughness, 0.5);
}

#[test]
fn the_skin_carries_its_inverse_bind_matrix() {
    let m = load();
    assert_eq!(m.skins.len(), 1);
    let skin = &m.skins[0];
    assert_eq!(skin.len(), 1);
    assert_eq!(skin.joints, vec![2]);
    // The second inverse bind matrix translates by x=1, so its last column
    // carries that in the fourth row of the column-major array.
    let m1 = skin.inverse_bind[1].to_cols_array();
    assert!((m1[12] - 1.0).abs() < 1e-6, "{m1:?}");
    assert!((m1[15] - 1.0).abs() < 1e-6, "{m1:?}");
}

#[test]
fn the_node_count_comes_through() {
    let m = load();
    assert_eq!(m.node_transforms.len(), 3);
}

#[test]
fn a_node_translation_becomes_a_world_transform() {
    let m = load();
    // Node 2 is the bone, translated to y = 1.
    let t = m.node_transform(2);
    let p = t * glam::Vec4::new(0.0, 0.0, 0.0, 1.0);
    assert!((p.y - 1.0).abs() < 1e-6, "{p:?}");
}

#[test]
fn the_animation_channel_comes_through() {
    let m = load();
    assert_eq!(m.clips.len(), 1);
    let clip = &m.clips[0];
    assert_eq!(clip.name, "lift");
    assert_eq!(clip.channels.len(), 1);
    assert_eq!(clip.channels[0].node, 2);
    assert_eq!(clip.channels[0].keyframes.len(), 2);
    assert!((clip.duration() - 1.0).abs() < 1e-6);
}

#[test]
fn the_animation_interpolates_between_its_keyframes() {
    let m = load();
    let ch = &m.clips[0].channels[0];
    match ch.sample(0.5) {
        ChannelValue::Translation(v) => {
            assert!(
                (v.y - 1.0).abs() < 1e-6,
                "halfway up a 0-to-2 lift is y=1: {v:?}"
            )
        }
        other => panic!("{other:?}"),
    }
}

#[test]
fn the_animation_reaches_its_final_value() {
    let m = load();
    let ch = &m.clips[0].channels[0];
    match ch.sample(1.0) {
        ChannelValue::Translation(v) => assert!((v.y - 2.0).abs() < 1e-6, "{v:?}"),
        other => panic!("{other:?}"),
    }
}

#[test]
fn playback_advances_through_the_loaded_clip() {
    let m = load();
    let mut p = Playback::new(0);
    p.loop_mode = LoopMode::Once;
    let clip = m.clip(0).unwrap();
    let at = p.advance(clip, 0.5);
    assert!((at - 0.5).abs() < 1e-6, "{at}");
}

#[test]
fn the_model_bounds_cover_the_triangle() {
    let m = load();
    let (lo, hi) = m.bounds().expect("a mesh has bounds");
    assert_eq!(lo, glam::Vec3::new(0.0, 0.0, 0.0));
    assert_eq!(hi, glam::Vec3::new(1.0, 1.0, 0.0));
}

#[test]
fn the_longest_clip_is_the_only_one() {
    let m = load();
    assert_eq!(m.longest_clip(), Some(0));
}

#[test]
fn loading_from_bytes_matches_loading_from_a_file() {
    let bytes = std::fs::read(fixture()).expect("the fixture exists");
    let from_file = load();
    let from_memory = vibe_gltf::load_from_bytes(&bytes, Some(&fixture())).expect("loads");
    assert_eq!(from_file.vertex_count(), from_memory.vertex_count());
    assert_eq!(from_file.triangle_count(), from_memory.triangle_count());
    assert_eq!(from_file.clips.len(), from_memory.clips.len());
    assert_eq!(from_file.skins.len(), from_memory.skins.len());
}

#[test]
fn garbage_bytes_are_a_parse_error() {
    let err = vibe_gltf::load_from_bytes(b"not a glb at all", None).unwrap_err();
    assert!(matches!(err, vibe_gltf::GltfError::Parse { .. }), "{err:?}");
}

#[test]
fn a_truncated_document_is_a_parse_error() {
    let bytes = std::fs::read(fixture()).unwrap();
    // Cutting the file short must be reported, not read past the end.
    let err = vibe_gltf::load_from_bytes(&bytes[..40], None).unwrap_err();
    assert!(matches!(err, vibe_gltf::GltfError::Parse { .. }), "{err:?}");
}
