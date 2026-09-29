//! Graphics pipelines, vertex layouts and dynamic rendering.

pub mod error;
pub mod layout;
pub mod render_pass;
pub mod state;

pub use error::PipelineError;
pub use layout::{Attribute, VertexFormat, VertexLayout, vertex_layout_for_quad};
pub use render_pass::{ColorAttachment, RenderPassDesc, begin_rendering_info};
pub use state::{BlendState, DepthState, PipelineDesc, PrimitiveState, RasterState, SampleState};
