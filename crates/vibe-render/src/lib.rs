//! The 2D batched quad renderer.

pub mod batch;
pub mod bind_group;
pub mod error;
pub mod mesh3d;
pub mod pipeline;
pub mod renderer;
pub mod texture;
pub mod ui;

pub use batch::Batcher;
pub use bind_group::{BindGroup, BindGroupDesc, create_sampler};
pub use error::RenderError;
pub use pipeline::{QUAD_SHADER, QuadPipeline, build_quad_pipeline};
pub use renderer::{QuadRenderer, RendererDesc};
