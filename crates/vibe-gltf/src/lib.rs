//! glTF/GLB loading, skeletal animation and sprite sheets for vibeEngine.
//!
//! The crate is pure logic: it turns files into the engine's mesh, skin, clip
//! and sheet types and never touches the GPU. That is deliberate, because it
//! means an asset bug shows up as a failing test rather than as a blank model.

pub mod anim;
pub mod error;
pub mod loader;
pub mod mesh;
pub mod sprite;
pub mod text;

pub use anim::{
    Channel, ChannelPath, ChannelValue, Clip, Interpolation, Keyframe, LoopMode, Playback,
    bone_matrices, compose_local,
};
pub use error::GltfError;
pub use loader::{Image, MAX_JOINTS, Model, from_document, load, load_from_bytes};
pub use mesh::{
    Material, Mesh, MeshVertex, Skin, TextureRef, UvSet, VERTEX_STRIDE, dominant_joint,
    make_vertex, normalize_uv, normalize_weights, skin_matrices,
};
pub use sprite::{
    FrameRect, SheetClock, SheetDesc, SheetLayout, grid_frames, slice, strip_frames_horizontal,
    strip_frames_vertical, validate_sheet,
};
pub use text::{
    AtlasRect, CHANNELS, EM_DISTANCE_RANGE, GlyphEntry, GlyphRun, MsdfAtlas, MsdfFont, PlacedGlyph,
    TextAlign, TextDirection, align_run, layout, layout_with_direction,
};
