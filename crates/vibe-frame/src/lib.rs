//! Command buffers, per-frame sync objects and queue submission.

pub mod error;
pub mod pool;
pub mod submit;

pub use error::FrameError;
pub use pool::{CommandPool, FrameConfig, FrameResources};
pub use submit::{
    Submission, SubmitBatch, SubmitSync, TimelineValue, build_submit_info, build_wait_info,
};
