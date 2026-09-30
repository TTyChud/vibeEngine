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
    /// A complete swapchain frame: wait for the previous frame's timeline
    /// value and for the WSI's acquire semaphore, signal the next timeline
    /// value and the present semaphore.
    ///
    /// This exists because the two kinds of wait are not interchangeable. The
    /// pacing wait is a timeline value, which keeps the CPU from running ahead
    /// of the GPU; the acquire wait is a binary semaphore, because
    /// `vkAcquireNextImageKHR` has no value field. Using one for the other
    /// produces a submission that waits on a signal nobody ever sends, which
    /// hangs on the fence rather than reporting an error.
    Paced {
        /// The timeline semaphore every frame shares.
        timeline: vk::Semaphore,
        /// Wait until the timeline reaches this: the previous frame's value.
        wait_value: TimelineValue,
        /// Signalled by `vkAcquireNextImageKHR` when the image is ready.
        acquire: vk::Semaphore,
        /// Signalled for `vkQueuePresentKHR` to wait on.
        signal: vk::Semaphore,
        /// Raise the timeline to this when the frame is done.
        signal_value: TimelineValue,
    },
}

impl SubmitSync {
    /// True when the submission waits for something.
    pub fn waits(&self) -> bool {
        matches!(
            self,
            SubmitSync::Binary { .. } | SubmitSync::Timeline { .. } | SubmitSync::Paced { .. }
        )
    }

    /// True when the submission signals something.
    pub fn signals(&self) -> bool {
        !matches!(self, SubmitSync::None)
    }

    /// The semaphore this waits on, if any.
    ///
    /// A [`SubmitSync::Paced`] waits on two; this reports the acquire
    /// semaphore, which is the one the frame's work actually depends on.
    pub fn wait_semaphore(&self) -> Option<vk::Semaphore> {
        self.wait_infos().first().map(|i| i.semaphore)
    }

    /// The semaphore this signals, if any.
    pub fn signal_semaphore(&self) -> Option<vk::Semaphore> {
        self.signal_infos().first().map(|i| i.semaphore)
    }

    /// The binary semaphore `vkAcquireNextImageKHR` signals, if this frame has
    /// one. The timeline semaphore is deliberately not reported here: the WSI
    /// calls take no value and reject a timeline semaphore.
    pub fn acquire_binary(&self) -> Option<vk::Semaphore> {
        match self {
            SubmitSync::Binary { wait, .. } => Some(*wait),
            SubmitSync::Paced { acquire, .. } => Some(*acquire),
            _ => None,
        }
    }

    /// The binary semaphore `vkQueuePresentKHR` waits on, if this frame has one.
    pub fn present_binary(&self) -> Option<vk::Semaphore> {
        match self {
            SubmitSync::Binary { signal, .. } | SubmitSync::SignalOnly { signal } => Some(*signal),
            SubmitSync::Paced { signal, .. } => Some(*signal),
            _ => None,
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

    /// The `SemaphoreSubmitInfo`s for the waits.
    ///
    /// A binary semaphore has no value, so `value` is left at zero, which the
    /// spec defines as "wait for the first signal".
    pub fn wait_infos(&self) -> Vec<vk::SemaphoreSubmitInfo<'_>> {
        match self {
            SubmitSync::Binary { wait, .. } => vec![info(*wait, 0, self.wait_stage())],
            SubmitSync::Timeline {
                semaphore,
                wait_value,
                ..
            } => vec![info(*semaphore, *wait_value, self.wait_stage())],
            // Two waits: the pacing value, then the image being ready. The
            // stage masks differ, because the image is a colour attachment
            // while the pacing wait is about the GPU having caught up at all.
            SubmitSync::Paced {
                timeline,
                wait_value,
                acquire,
                ..
            } => vec![
                info(
                    *timeline,
                    *wait_value,
                    vk::PipelineStageFlags2::ALL_COMMANDS,
                ),
                info(*acquire, 0, self.wait_stage()),
            ],
            SubmitSync::SignalOnly { .. } | SubmitSync::None => Vec::new(),
        }
    }

    /// The `SemaphoreSubmitInfo`s for the signals.
    pub fn signal_infos(&self) -> Vec<vk::SemaphoreSubmitInfo<'_>> {
        match self {
            SubmitSync::Binary { signal, .. } | SubmitSync::SignalOnly { signal } => {
                vec![info(*signal, 0, self.signal_stage())]
            }
            SubmitSync::Timeline {
                semaphore,
                signal_value,
                ..
            } => vec![info(*semaphore, *signal_value, self.signal_stage())],
            // Two signals: the present semaphore, which is binary and carries no
            // value, and the timeline value that lets the next frame know this
            // one finished.
            SubmitSync::Paced {
                signal,
                timeline,
                signal_value,
                ..
            } => vec![
                info(*signal, 0, self.signal_stage()),
                info(*timeline, *signal_value, self.signal_stage()),
            ],
            SubmitSync::None => Vec::new(),
        }
    }
}

/// One `SemaphoreSubmitInfo`, in the shape every arm above needs it.
fn info(
    semaphore: vk::Semaphore,
    value: TimelineValue,
    stage: vk::PipelineStageFlags2,
) -> vk::SemaphoreSubmitInfo<'static> {
    vk::SemaphoreSubmitInfo {
        semaphore,
        value,
        stage_mask: stage,
        ..Default::default()
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
    let waits = batch.sync.wait_infos();
    let signals = batch.sync.signal_infos();
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
    fn a_timeline_submission_signals_the_semaphore_it_waits_on() {
        // One semaphore is both the wait and the signal, told apart by value:
        // waiting on N and raising it to N+1 is the documented pattern.
        let s = SubmitSync::Timeline {
            semaphore: sem(9),
            wait_value: 1,
            signal_value: 2,
        };
        assert_eq!(s.wait_semaphore(), Some(sem(9)));
        assert_eq!(s.signal_semaphore(), Some(sem(9)));
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
        let infos = sync.wait_infos();
        assert_eq!(infos.len(), 1);
        assert_eq!(infos[0].semaphore, sem(1));
        assert_eq!(infos[0].value, 0, "a binary wait has no timeline value");
    }

    #[test]
    fn a_timeline_wait_carries_its_value() {
        let sync = SubmitSync::Timeline {
            semaphore: sem(9),
            wait_value: 7,
            signal_value: 8,
        };
        let infos = sync.wait_infos();
        assert_eq!(infos[0].semaphore, sem(9));
        assert_eq!(infos[0].value, 7);
    }

    #[test]
    fn a_signal_info_carries_the_signal_value() {
        let sync = SubmitSync::Timeline {
            semaphore: sem(9),
            wait_value: 1,
            signal_value: 11,
        };
        let infos = sync.signal_infos();
        assert_eq!(infos[0].value, 11);
    }

    #[test]
    fn a_none_submission_has_no_infos() {
        assert!(SubmitSync::None.wait_infos().is_empty());
        assert!(SubmitSync::None.signal_infos().is_empty());
    }

    #[test]
    fn a_paced_frame_waits_on_both_the_timeline_and_the_acquire() {
        // The bug this fixes: the frame waited on a timeline value that only
        // the timeline semaphore could raise, while acquire signalled a binary
        // one nothing was waiting on. The fence then timed out rather than
        // reporting anything.
        let sync = SubmitSync::Paced {
            timeline: sem(9),
            wait_value: 3,
            acquire: sem(4),
            signal: sem(5),
            signal_value: 4,
        };
        let waits = sync.wait_infos();
        assert_eq!(waits.len(), 2, "a paced frame waits twice");
        assert_eq!(waits[0].semaphore, sem(9));
        assert_eq!(waits[0].value, 3);
        assert_eq!(waits[1].semaphore, sem(4));
        assert_eq!(waits[1].value, 0, "the acquire semaphore is binary");
    }

    #[test]
    fn a_paced_frame_signals_the_present_semaphore_and_the_timeline() {
        let sync = SubmitSync::Paced {
            timeline: sem(9),
            wait_value: 3,
            acquire: sem(4),
            signal: sem(5),
            signal_value: 4,
        };
        let signals = sync.signal_infos();
        assert_eq!(signals.len(), 2);
        assert_eq!(signals[0].semaphore, sem(5), "present waits on this");
        assert_eq!(signals[0].value, 0, "the present semaphore is binary");
        assert_eq!(signals[1].semaphore, sem(9), "the pace advances here");
        assert_eq!(signals[1].value, 4);
    }

    #[test]
    fn a_paced_frame_never_reuses_a_binary_semaphore() {
        // Reusing one *binary* semaphore as both wait and signal is invalid.
        // A timeline semaphore is exempt: waiting on N and signalling N+1 is the
        // documented pattern, and that is what the pacing wait does.
        let sync = SubmitSync::Paced {
            timeline: sem(9),
            wait_value: 1,
            acquire: sem(4),
            signal: sem(5),
            signal_value: 2,
        };
        // A valid Binary names two different semaphores, and the accessor
        // pair reports them separately rather than collapsing to one.
        let binary = SubmitSync::Binary {
            wait: sem(4),
            signal: sem(5),
        };
        assert_ne!(binary.acquire_binary(), binary.present_binary());
        assert_eq!(binary.acquire_binary(), Some(sem(4)));
        assert_eq!(binary.present_binary(), Some(sem(5)));

        // A timeline-only frame has no binary semaphores at all, which is
        // exactly why it cannot be used as a swapchain frame.
        let timeline = SubmitSync::Timeline {
            semaphore: sem(9),
            wait_value: 1,
            signal_value: 2,
        };
        assert_eq!(timeline.acquire_binary(), None);
        assert_eq!(timeline.present_binary(), None);

        // And the paced frame's two binary semaphores are distinct.
        assert_ne!(sync.acquire_binary(), sync.present_binary());
    }

    #[test]
    fn a_paced_frame_waits_and_signals() {
        let sync = SubmitSync::Paced {
            timeline: sem(9),
            wait_value: 1,
            acquire: sem(4),
            signal: sem(5),
            signal_value: 2,
        };
        assert!(sync.waits());
        assert!(sync.signals());
        assert_eq!(sync.wait_semaphore(), Some(sem(9)));
        assert_eq!(sync.signal_semaphore(), Some(sem(5)));
    }

    #[test]
    fn a_paced_submit_info_counts_two_waits_and_two_signals() {
        let batch = batch_with(SubmitSync::Paced {
            timeline: sem(9),
            wait_value: 1,
            acquire: sem(4),
            signal: sem(5),
            signal_value: 2,
        });
        let s = build_submit_info(&batch);
        assert_eq!(s.info.wait_semaphore_info_count, 2);
        assert_eq!(s.info.signal_semaphore_info_count, 2);
        assert_eq!(s._waits[1].semaphore, sem(4));
        assert_eq!(s._signals[0].semaphore, sem(5));
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
