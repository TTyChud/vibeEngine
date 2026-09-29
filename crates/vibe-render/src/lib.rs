//! The 2D batched quad renderer.

pub mod batch;
pub mod error;
pub mod pipeline;
pub mod renderer;

pub use batch::Batcher;
pub use error::RenderError;
pub use pipeline::{QUAD_SHADER, QuadPipeline, build_quad_pipeline};
pub use renderer::{QuadRenderer, RendererDesc};
