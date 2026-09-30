//! What can go wrong while loading a glTF file.

use std::path::PathBuf;

use thiserror::Error;

/// A failure in asset loading.
#[derive(Debug, Error)]
pub enum GltfError {
    /// The file could not be read.
    #[error("cannot read {path}: {source}")]
    Io {
        /// The file being read.
        path: PathBuf,
        /// The underlying failure.
        #[source]
        source: std::io::Error,
    },

    /// The bytes are not a glTF or GLB document.
    #[error("cannot parse {path} as glTF: {source}")]
    Parse {
        /// The file being parsed.
        path: PathBuf,
        /// The parser's message.
        #[source]
        source: gltf::Error,
    },

    /// A mesh primitive asks for an attribute the loader does not produce.
    #[error("mesh {mesh} primitive {primitive} is missing attribute {attribute}")]
    MissingAttribute {
        /// Index of the mesh.
        mesh: usize,
        /// Index of the primitive inside the mesh.
        primitive: usize,
        /// The missing attribute, such as `POSITION`.
        attribute: &'static str,
    },

    /// An accessor is used with a type the loader cannot read.
    #[error(
        "accessor {accessor} has {count} component(s) of {kind}, which is not a scalar, vec2, vec3 or vec4"
    )]
    UnsupportedAccessor {
        /// Index of the accessor.
        accessor: usize,
        /// How many components each element has.
        count: usize,
        /// The component type's name.
        kind: &'static str,
    },

    /// An index buffer references a vertex past the end of its vertex buffer.
    #[error("index {index} in {mesh}/{primitive} is past the {count} vertices available")]
    IndexOutOfRange {
        /// Index of the mesh.
        mesh: usize,
        /// Index of the primitive.
        primitive: usize,
        /// The offending index.
        index: usize,
        /// How many vertices the primitive has.
        count: usize,
    },

    /// A skin names a joint the skeleton does not contain.
    #[error("skin {skin} names joint {joint}, but the skeleton has {count}")]
    UnknownJoint {
        /// Index of the skin.
        skin: usize,
        /// The offending joint index.
        joint: usize,
        /// How many joints the skeleton has.
        count: usize,
    },

    /// A skin has more joints than the engine's bone budget allows.
    #[error("skin {skin} has {joints} joints, over the {limit}-bone limit")]
    TooManyJoints {
        /// Index of the skin.
        skin: usize,
        /// How many joints the skin declares.
        joints: usize,
        /// The engine's limit.
        limit: usize,
    },

    /// An animation sampler reads a channel the loader does not support.
    #[error("animation {animation} sampler {sampler} targets unsupported path {path:?}")]
    UnsupportedPath {
        /// Index of the animation.
        animation: usize,
        /// Index of the sampler.
        sampler: usize,
        /// The glTF animation channel path.
        path: String,
    },

    /// A sprite sheet names a frame the source rectangle does not fit inside.
    #[error("frame {frame} of {name} is {rect}, which does not fit in {width}x{height}")]
    FrameOutOfBounds {
        /// The sheet's name.
        name: String,
        /// Index of the frame.
        frame: usize,
        /// The frame rectangle.
        rect: String,
        /// The sheet's width.
        width: u32,
        /// The sheet's height.
        height: u32,
    },

    /// A sprite sheet has no frames, or a frame count of zero.
    #[error("sprite sheet {name} declares {frames} frame(s); at least one is required")]
    NoFrames {
        /// The sheet's name.
        name: String,
        /// How many frames were declared.
        frames: usize,
    },

    /// An embedded image could not be decoded.
    #[error("cannot decode image {name}: {message}")]
    Decode {
        /// The image's name.
        name: String,
        /// The decoder's message.
        message: String,
    },
}
