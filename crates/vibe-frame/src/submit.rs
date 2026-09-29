//! Building a queue submission from a frame's recorded work.
//!
//! Vulkan 1.3's `vkQueueSubmit2` describes semaphores with a
//! `SemaphoreSubmitInfo` carrying a stage mask and a value, so a binary wait and
//! a timeline wait differ only in whether `value` is set. That is why the 1.3
//! path is used for both: one struct, one code path, and no separate
//! `p_wait_semaphore_values` array to get wrong.

use ash::vk;

/// A value on a timeline semaphore.
pub type TimelineValue = u64;

/// The semaphores and value a submission waits on and signals.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubmitSync {
    /// No wait and no signal, for a frame that needs neither.
    None,
    /// Wait on a binary semaphore, signal another.
    Binary {
        /// Signalled by the swapchain when the image is ready.
        wait: vk::Semaphore,
        /// Signalled when this frame's rendering is done.
        signal: vk::Semaphore,
    },
    /// Wait on and signal one timeline semaphore by value.
    ///
    /// A frame waits for the previous frame's value and signals the next, which
    /// is what lets the CPU run ahead without consuming a semaphore per frame.
    Timeline {
        /// The semaphore shared by every frame.
        semaphore: vk::Semaphore,
        /// Wait until the timeline reaches this.
        wait_value: TimelineValue,
        /// Raise the timeline to this when the frame is done.
        signal_value: TimelineValue,
    },
    /// Signal only, for a frame that must not wait.
    SignalOnly {
        /// Signalled when this frame's rendering is done.
        signal: vk::Semaphore,
    },
}

impl SubmitSync {
    /// True when the submission waits for something.
    pub fn waits(&self) -> bool {
        matches!(
            self,
            SubmitSync::Binary { .. } | SubmitSync::Timeline { .. }
        )
    }

    /// True when the submission signals something.
    pub fn signals(&self) -> bool {
        !matches!(self, SubmitSync::None)
    }

    /// The semaphore this waits on, if any.
    pub fn wait_semaphore(&self) -> Option<vk::Semaphore> {
        match self {
            SubmitSync::Binary { wait, .. } => Some(*wait),
            SubmitSync::Timeline { semaphore, .. } => Some(*semaphore),
            _ => None,
        }
    }

    /// The semaphore this signals, if any.
    pub fn signal_semaphore(&self) -> Option<vk::Semaphore> {
        match self {
            SubmitSync::Binary { signal, .. } => Some(*signal),
            SubmitSync::SignalOnly { signal } => Some(*signal),
            SubmitSync::None => None,
            SubmitSync::Timeline { .. } => None,
        }
    }

    /// The stage a wait blocks at.
    ///
    /// Acquiring a swapchain image is a colour-attachment operation, so the
    /// wait belongs there rather than at the top of the pipe.
    pub fn wait_stage(&self) -> vk::PipelineStageFlags2 {
        if self.waits() {
            vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT
        } else {
            vk::PipelineStageFlags2::NONE
        }
    }

    /// The stage a signal is visible from.
    pub fn signal_stage(&self) -> vk::PipelineStageFlags2 {
        if self.signals() {
            vk::PipelineStageFlags2::ALL_COMMANDS
        } else {
            vk::PipelineStageFlags2::NONE
        }
    }

    /// The `SemaphoreSubmitInfo` for the wait, if there is one.
    ///
    /// A binary semaphore has no value, so `value` is left at zero, which the
    /// spec defines as "wait for the first signal".
    pub fn wait_info(&self) -> Option<vk::SemaphoreSubmitInfo<'_>> {
        let (semaphore, value) = match self {
            SubmitSync::Binary { wait, .. } => (*wait, 0),
            SubmitSync::Timeline {
                semaphore,
                wait_value,
                ..
            } => (*semaphore, *wait_value),
            _ => return None,
        };
        Some(vk::SemaphoreSubmitInfo {
            semaphore,
            value,
            stage_mask: self.wait_stage(),
            ..Default::default()
        })
    }

    /// The `SemaphoreSubmitInfo` for the signal, if there is one.
    pub fn signal_info(&self) -> Option<vk::SemaphoreSubmitInfo<'_>> {
        let (semaphore, value) = match self {
            SubmitSync::Binary { signal, .. } | SubmitSync::SignalOnly { signal } => (*signal, 0),
            SubmitSync::Timeline {
                semaphore,
                signal_value,
                ..
            } => (*semaphore, *signal_value),
            SubmitSync::None => return None,
        };
        Some(vk::SemaphoreSubmitInfo {
            semaphore,
            value,
            stage_mask: self.signal_stage(),
            ..Default::default()
        })
    }
}

/// The command buffers and sync for one submission.
#[derive(Debug, Clone, PartialEq)]
pub struct SubmitBatch {
    /// Command buffers to execute, in order.
    pub command_buffers: Vec<vk::CommandBuffer>,
    /// The wait and signal for this submission.
    pub sync: SubmitSync,
}

impl SubmitBatch {
    /// A batch of one command buffer.
    pub fn single(buffer: vk::CommandBuffer, sync: SubmitSync) -> SubmitBatch {
        SubmitBatch {
            command_buffers: vec![buffer],
            sync,
        }
    }

    /// Add another command buffer to the same submission.
    ///
    /// Batching them means one `vkQueueSubmit2` instead of several, which is
    /// worth doing when the barriers between them are already recorded.
    pub fn push(&mut self, buffer: vk::CommandBuffer) {
        self.command_buffers.push(buffer);
    }

    /// True when there is nothing to submit.
    pub fn is_empty(&self) -> bool {
        self.command_buffers.is_empty()
    }
}

/// The `SubmitInfo2` for a batch, with the descriptor arrays it points at.
///
/// The arrays are returned alongside the info because `SubmitInfo2` holds raw
/// pointers into them: they must outlive the `vkQueueSubmit2` call but must not
/// outlive this tuple, or the info dangles. Keeping them in one struct makes
/// that impossible to get wrong.
pub struct Submission<'a> {
    /// The info to hand to `vkQueueSubmit2`.
    pub info: vk::SubmitInfo2<'a>,
    /// Kept alive for the pointers in `info`.
    _waits: Vec<vk::SemaphoreSubmitInfo<'a>>,
    _signals: Vec<vk::SemaphoreSubmitInfo<'a>>,
    _buffers: Vec<vk::CommandBufferSubmitInfo<'a>>,
}

/// Build a `SubmitInfo2` and the arrays it references.
pub fn build_submit_info(batch: &SubmitBatch) -> Submission<'_> {
    let waits: Vec<_> = batch.sync.wait_info().into_iter().collect();
    let signals: Vec<_> = batch.sync.signal_info().into_iter().collect();
    let buffers: Vec<_> = batch
        .command_buffers
        .iter()
        .map(|b| vk::CommandBufferSubmitInfo {
            command_buffer: *b,
            ..Default::default()
        })
        .collect();

    let info = vk::SubmitInfo2 {
        wait_semaphore_info_count: waits.len() as u32,
        p_wait_semaphore_infos: if waits.is_empty() {
            std::ptr::null()
        } else {
            waits.as_ptr()
        },
        command_buffer_info_count: buffers.len() as u32,
        p_command_buffer_infos: if buffers.is_empty() {
            std::ptr::null()
        } else {
            buffers.as_ptr()
        },
        signal_semaphore_info_count: signals.len() as u32,
        p_signal_semaphore_infos: if signals.is_empty() {
            std::ptr::null()
        } else {
            signals.as_ptr()
        },
        ..Default::default()
    };

    Submission {
        info,
        _waits: waits,
        _signals: signals,
        _buffers: buffers,
    }
}

/// A `SemaphoreWaitInfo` for a timeline wait.
pub fn build_wait_info<'a>(
    semaphores: &'a [vk::Semaphore],
    values: &'a [TimelineValue],
) -> vk::SemaphoreWaitInfo<'a> {
    vk::SemaphoreWaitInfo {
        semaphore_count: semaphores.len() as u32,
        p_semaphores: semaphores.as_ptr(),
        p_values: values.as_ptr(),
        ..Default::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ash::vk::Handle as _;

    fn sem(n: u64) -> vk::Semaphore {
        vk::Semaphore::from_raw(n)
    }

    fn batch_with(sync: SubmitSync) -> SubmitBatch {
        SubmitBatch::single(vk::CommandBuffer::from_raw(1), sync)
    }

    #[test]
    fn a_binary_submission_waits_and_signals() {
        let s = SubmitSync::Binary {
            wait: sem(1),
            signal: sem(2),
        };
        assert!(s.waits());
        assert!(s.signals());
        assert_eq!(s.wait_semaphore(), Some(sem(1)));
        assert_eq!(s.signal_semaphore(), Some(sem(2)));
    }

    #[test]
    fn a_timeline_submission_waits_on_a_value() {
        let s = SubmitSync::Timeline {
            semaphore: sem(9),
            wait_value: 3,
            signal_value: 4,
        };
        assert!(s.waits());
        assert!(s.signals());
        assert_eq!(s.wait_semaphore(), Some(sem(9)));
    }

    #[test]
    fn a_timeline_submission_signals_no_separate_semaphore() {
        let s = SubmitSync::Timeline {
            semaphore: sem(9),
            wait_value: 1,
            signal_value: 2,
        };
        assert_eq!(
            s.signal_semaphore(),
            None,
            "one semaphore is both wait and signal"
        );
    }

    #[test]
    fn a_signal_only_submission_does_not_wait() {
        let s = SubmitSync::SignalOnly { signal: sem(3) };
        assert!(!s.waits());
        assert!(s.signals());
        assert_eq!(s.wait_semaphore(), None);
    }

    #[test]
    fn a_none_submission_does_nothing() {
        let s = SubmitSync::None;
        assert!(!s.waits());
        assert!(!s.signals());
        assert_eq!(s.signal_semaphore(), None);
    }

    #[test]
    fn a_wait_blocks_at_colour_attachment_output() {
        assert_eq!(
            SubmitSync::Binary {
                wait: sem(1),
                signal: sem(2)
            }
            .wait_stage(),
            vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT
        );
    }

    #[test]
    fn a_signal_is_visible_from_all_commands() {
        assert_eq!(
            SubmitSync::Binary {
                wait: sem(1),
                signal: sem(2)
            }
            .signal_stage(),
            vk::PipelineStageFlags2::ALL_COMMANDS
        );
    }

    #[test]
    fn a_submission_with_no_sync_has_no_stages() {
        assert_eq!(SubmitSync::None.wait_stage(), vk::PipelineStageFlags2::NONE);
        assert_eq!(
            SubmitSync::None.signal_stage(),
            vk::PipelineStageFlags2::NONE
        );
    }

    #[test]
    fn a_batch_starts_with_one_buffer() {
        let b = batch_with(SubmitSync::None);
        assert_eq!(b.command_buffers.len(), 1);
        assert!(!b.is_empty());
    }

    #[test]
    fn pushing_buffers_batches_them() {
        let mut b = batch_with(SubmitSync::None);
        b.push(vk::CommandBuffer::from_raw(2));
        b.push(vk::CommandBuffer::from_raw(3));
        assert_eq!(b.command_buffers.len(), 3);
    }

    #[test]
    fn an_empty_batch_reports_it() {
        let b = SubmitBatch {
            command_buffers: vec![],
            sync: SubmitSync::None,
        };
        assert!(b.is_empty());
    }

    #[test]
    fn a_binary_wait_carries_no_value() {
        let sync = SubmitSync::Binary {
            wait: sem(1),
            signal: sem(2),
        };
        let info = sync.wait_info().unwrap();
        assert_eq!(info.semaphore, sem(1));
        assert_eq!(info.value, 0, "a binary wait has no timeline value");
    }

    #[test]
    fn a_timeline_wait_carries_its_value() {
        let sync = SubmitSync::Timeline {
            semaphore: sem(9),
            wait_value: 7,
            signal_value: 8,
        };
        let info = sync.wait_info().unwrap();
        assert_eq!(info.semaphore, sem(9));
        assert_eq!(info.value, 7);
    }

    #[test]
    fn a_signal_info_carries_the_signal_value() {
        let sync = SubmitSync::Timeline {
            semaphore: sem(9),
            wait_value: 1,
            signal_value: 11,
        };
        let info = sync.signal_info().unwrap();
        assert_eq!(info.value, 11);
    }

    #[test]
    fn a_none_submission_has_no_infos() {
        assert!(SubmitSync::None.wait_info().is_none());
        assert!(SubmitSync::None.signal_info().is_none());
    }

    #[test]
    fn a_binary_submit_info_counts_one_wait_and_one_signal() {
        let batch = batch_with(SubmitSync::Binary {
            wait: sem(1),
            signal: sem(2),
        });
        let s = build_submit_info(&batch);
        assert_eq!(s.info.wait_semaphore_info_count, 1);
        assert_eq!(s.info.signal_semaphore_info_count, 1);
        assert_eq!(s._waits[0].semaphore, sem(1));
        assert_eq!(s._signals[0].semaphore, sem(2));
    }

    #[test]
    fn a_timeline_submit_info_names_one_semaphore_twice() {
        let batch = batch_with(SubmitSync::Timeline {
            semaphore: sem(9),
            wait_value: 4,
            signal_value: 5,
        });
        let s = build_submit_info(&batch);
        assert_eq!(s._waits[0].semaphore, sem(9));
        assert_eq!(s._signals[0].semaphore, sem(9));
        assert_eq!(s._waits[0].value, 4);
        assert_eq!(s._signals[0].value, 5);
    }

    #[test]
    fn a_signal_only_submit_info_has_no_wait() {
        let batch = batch_with(SubmitSync::SignalOnly { signal: sem(3) });
        let s = build_submit_info(&batch);
        assert_eq!(s.info.wait_semaphore_info_count, 0);
        assert!(s.info.p_wait_semaphore_infos.is_null());
        assert_eq!(s.info.signal_semaphore_info_count, 1);
    }

    #[test]
    fn a_none_submit_info_has_no_sync_at_all() {
        let batch = batch_with(SubmitSync::None);
        let s = build_submit_info(&batch);
        assert_eq!(s.info.wait_semaphore_info_count, 0);
        assert_eq!(s.info.signal_semaphore_info_count, 0);
        assert!(s.info.p_wait_semaphore_infos.is_null());
        assert!(s.info.p_signal_semaphore_infos.is_null());
    }

    #[test]
    fn the_submit_info_counts_every_command_buffer() {
        let mut batch = batch_with(SubmitSync::None);
        batch.push(vk::CommandBuffer::from_raw(2));
        let s = build_submit_info(&batch);
        assert_eq!(s.info.command_buffer_info_count, 2);
        assert_eq!(s._buffers[1].command_buffer, vk::CommandBuffer::from_raw(2));
    }

    #[test]
    fn an_empty_batch_yields_null_buffers() {
        let batch = SubmitBatch {
            command_buffers: vec![],
            sync: SubmitSync::None,
        };
        let s = build_submit_info(&batch);
        assert_eq!(s.info.command_buffer_info_count, 0);
        assert!(s.info.p_command_buffer_infos.is_null());
    }

    #[test]
    fn a_wait_info_pairs_semaphores_with_values() {
        let semaphores = [sem(1), sem(2)];
        let values = [10u64, 20u64];
        let info = build_wait_info(&semaphores, &values);
        assert_eq!(info.semaphore_count, 2);
        assert_eq!(unsafe { *info.p_semaphores.add(1) }, sem(2));
        assert_eq!(unsafe { *info.p_values.add(1) }, 20);
    }

    #[test]
    fn an_empty_wait_info_is_null_safe() {
        let info = build_wait_info(&[], &[]);
        assert_eq!(info.semaphore_count, 0);
    }
}
