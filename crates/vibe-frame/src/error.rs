//! Errors from frame recording and submission.

impl From<ash::vk::Result> for FrameError {
    fn from(e: ash::vk::Result) -> Self {
        FrameError::Vk(e)
    }
}

/// A frame could not be recorded or submitted.
#[derive(Debug, thiserror::Error)]
pub enum FrameError {
    /// A command buffer was begun twice, or ended while already recording.
    #[error("command buffer {0} is not in a recordable state")]
    NotRecording(u32),
    /// The GPU is too far behind to submit another frame.
    #[error("all {0} frames in flight are still busy")]
    FramesBusy(usize),
    /// A call was made against a slot whose work has not finished.
    #[error("frame slot {slot} is still in flight, waited on {target}")]
    SlotBusy {
        /// The slot that was used too early.
        slot: usize,
        /// The value it is waiting for.
        target: u64,
    },
    /// A Vulkan call failed.
    #[error("vulkan call failed: {0:?}")]
    Vk(ash::vk::Result),
}
