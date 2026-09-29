//! Frame pacing: the sync primitives that differ per tier.
//!
//! Timeline semaphores let a frame wait for a specific value instead of
//! consuming a binary semaphore, which is what lets the CPU run ahead of the
//! GPU by a fixed number of frames. Below the Sync2Timeline tier the same role
//! is played by per-frame binary semaphores and fences, so the API here is
//! identical on all three tiers and the difference is an implementation detail.
//!
//! Selecting the right configuration and sizing the ring are pure logic and are
//! tested without a device; creating the Vulkan objects is not.

use ash::vk;

use crate::sync::SyncTier;

/// How many frames the CPU may run ahead of the GPU.
///
/// Two is the usual choice: one frame in flight being recorded, one being
/// submitted. A deeper queue hides more latency but makes input feel further
/// away, and it costs memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrameConfig {
    /// Frames in flight, at least 1.
    pub frames_in_flight: usize,
    /// Whether to use timeline semaphores rather than binary ones.
    pub use_timeline: bool,
    /// Whether to block on a fence rather than a semaphore between frames.
    pub use_fences: bool,
}

impl FrameConfig {
    /// A configuration for a device on the given tier.
    ///
    /// Timeline semaphores need Sync2Timeline; anything below gets binary
    /// semaphores, and fences are enabled on every tier so a frame can wait on
    /// its own submission.
    pub fn for_tier(tier: SyncTier) -> FrameConfig {
        FrameConfig {
            frames_in_flight: default_frames_in_flight(),
            use_timeline: tier.supports_timeline(),
            use_fences: true,
        }
    }

    /// A configuration with an explicit frame count.
    ///
    /// # Panics
    ///
    /// Panics if `frames_in_flight` is zero, which would deadlock the wait.
    pub fn with_frames(tier: SyncTier, frames_in_flight: usize) -> FrameConfig {
        assert!(frames_in_flight > 0, "frames_in_flight must be at least 1");
        FrameConfig {
            frames_in_flight,
            ..FrameConfig::for_tier(tier)
        }
    }

    /// Number of semaphores to create: one per frame on the timeline tier,
    /// none otherwise, since a timeline semaphore is a single object.
    pub fn semaphores(&self) -> usize {
        if self.use_timeline {
            1
        } else {
            self.frames_in_flight
        }
    }

    /// Number of fences to create: one per frame when fences are enabled.
    pub fn fences(&self) -> usize {
        if self.use_fences {
            self.frames_in_flight
        } else {
            0
        }
    }

    /// Number of command buffers to allocate: one per frame.
    pub fn command_buffers(&self) -> usize {
        self.frames_in_flight
    }

    /// Bytes of per-frame uniforms or push-constant blocks, for budgeting.
    pub fn per_frame_bytes(&self, bytes_per_frame: usize) -> usize {
        bytes_per_frame * self.frames_in_flight
    }
}

/// The default frames in flight, capped to 3 so a slow machine does not build
/// up an unbounded queue.
pub const fn default_frames_in_flight() -> usize {
    2
}

/// How many frames the CPU may be ahead of the GPU.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FramePacer {
    /// Total frames submitted since start.
    pub submitted: u64,
    /// The value a timeline semaphore has reached.
    pub completed: u64,
    frames_in_flight: usize,
    use_timeline: bool,
}

impl FramePacer {
    /// A pacer for a configuration.
    pub fn new(config: FrameConfig) -> FramePacer {
        FramePacer {
            submitted: 0,
            completed: 0,
            frames_in_flight: config.frames_in_flight,
            use_timeline: config.use_timeline,
        }
    }

    /// Record that a frame was submitted, returning the slot it occupies.
    ///
    /// The slot is `submitted % frames_in_flight`, and a caller must not reuse
    /// a slot whose fence is still unsignalled.
    pub fn submit(&mut self) -> usize {
        let slot = (self.submitted % self.frames_in_flight as u64) as usize;
        self.submitted += 1;
        slot
    }

    /// Record that a frame's GPU work finished.
    pub fn complete(&mut self) {
        self.completed += 1;
    }

    /// True while the GPU is behind the allowed depth.
    pub fn is_busy(&self) -> bool {
        self.submitted.saturating_sub(self.completed) >= self.frames_in_flight as u64
    }

    /// Frames the CPU is ahead of the GPU.
    pub fn in_flight(&self) -> u64 {
        self.submitted.saturating_sub(self.completed)
    }

    /// The value to wait on for a timeline semaphore.
    ///
    /// On the binary path there is no value, so this returns `None` and the
    /// caller waits on a fence instead.
    pub fn timeline_target(&self) -> Option<u64> {
        self.use_timeline.then_some(self.submitted)
    }

    /// The slot a frame should wait on before reusing it.
    pub fn wait_slot(&self) -> usize {
        (self.completed % self.frames_in_flight as u64) as usize
    }

    /// Drop every frame in flight, e.g. after a device loss.
    pub fn reset(&mut self) {
        self.submitted = 0;
        self.completed = 0;
    }

    /// True when this pacer uses timeline semaphores.
    pub fn uses_timeline(&self) -> bool {
        self.use_timeline
    }
}

/// Deferred destruction: objects that cannot be freed until the GPU has passed
/// the submission that used them.
///
/// This is what keeps the renderer from freeing a frame's command buffer while
/// the GPU is still reading it, and it is tier-independent: the object carries
/// the fence or timeline value to wait for, and the caller decides how.
#[derive(Debug, Default)]
pub struct DeferredDestructionQueue {
    /// Objects waiting on a fence, as (fence, value, object).
    fenced: Vec<(vk::Fence, u64, u64)>,
    /// Objects waiting on a timeline value.
    timeline: Vec<(u64, u64)>,
}

impl DeferredDestructionQueue {
    /// An empty queue.
    pub fn new() -> DeferredDestructionQueue {
        DeferredDestructionQueue::default()
    }

    /// Queue an object to be freed once `fence` reaches `value`.
    pub fn push_fenced(&mut self, fence: vk::Fence, value: u64, object: u64) {
        self.fenced.push((fence, value, object));
    }

    /// Queue an object to be freed once the timeline reaches `value`.
    pub fn push_timeline(&mut self, value: u64, object: u64) {
        self.timeline.push((value, object));
    }

    /// Remove and return every object whose wait condition is satisfied.
    ///
    /// `fence_signalled` reports whether a given fence has reached a given
    /// value, and `timeline_value` is the timeline's current value.
    pub fn collect_ready<F>(&mut self, timeline_value: u64, mut fence_signalled: F) -> Vec<u64>
    where
        F: FnMut(vk::Fence, u64) -> bool,
    {
        let mut ready = Vec::new();

        self.timeline.retain(|(value, object)| {
            if timeline_value >= *value {
                ready.push(*object);
                false
            } else {
                true
            }
        });

        self.fenced.retain(|(fence, value, object)| {
            if fence_signalled(*fence, *value) {
                ready.push(*object);
                false
            } else {
                true
            }
        });

        ready
    }

    /// Number of objects still waiting.
    pub fn len(&self) -> usize {
        self.fenced.len() + self.timeline.len()
    }

    /// True when nothing is waiting.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Drop every queued object without freeing it, e.g. on device loss.
    pub fn clear(&mut self) {
        self.fenced.clear();
        self.timeline.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ash::vk::Handle as _;

    #[test]
    fn legacy_config_has_no_timeline() {
        let c = FrameConfig::for_tier(SyncTier::Legacy);
        assert!(!c.use_timeline);
        assert!(c.use_fences);
        assert_eq!(c.semaphores(), c.frames_in_flight);
    }

    #[test]
    fn sync2_has_no_timeline_but_sync2_timeline_does() {
        assert!(!FrameConfig::for_tier(SyncTier::Sync2).use_timeline);
        assert!(FrameConfig::for_tier(SyncTier::Sync2Timeline).use_timeline);
    }

    #[test]
    fn timeline_uses_a_single_semaphore() {
        let c = FrameConfig::for_tier(SyncTier::Sync2Timeline);
        assert_eq!(
            c.semaphores(),
            1,
            "a timeline semaphore is one object for all frames"
        );
    }

    #[test]
    fn binary_path_uses_one_semaphore_per_frame() {
        let c = FrameConfig::for_tier(SyncTier::Legacy);
        assert_eq!(c.semaphores(), c.frames_in_flight);
    }

    #[test]
    fn fences_and_command_buffers_are_per_frame() {
        let c = FrameConfig::with_frames(SyncTier::Sync2Timeline, 3);
        assert_eq!(c.fences(), 3);
        assert_eq!(c.command_buffers(), 3);
    }

    #[test]
    fn fences_can_be_disabled() {
        let mut c = FrameConfig::for_tier(SyncTier::Sync2Timeline);
        c.use_fences = false;
        assert_eq!(c.fences(), 0);
    }

    #[test]
    fn per_frame_bytes_scales_with_frames() {
        let c = FrameConfig::with_frames(SyncTier::Sync2, 4);
        assert_eq!(c.per_frame_bytes(1024), 4096);
    }

    #[test]
    fn default_frames_is_small() {
        assert!((1..=3).contains(&default_frames_in_flight()));
    }

    #[test]
    #[should_panic(expected = "at least 1")]
    fn zero_frames_panics() {
        FrameConfig::with_frames(SyncTier::Sync2, 0);
    }

    #[test]
    fn pacer_cycles_slots() {
        let mut p = FramePacer::new(FrameConfig::with_frames(SyncTier::Sync2Timeline, 2));
        assert_eq!(p.submit(), 0);
        assert_eq!(p.submit(), 1);
        assert_eq!(p.submit(), 0, "slots must wrap at the frame count");
    }

    #[test]
    fn pacer_is_not_busy_before_the_depth_is_reached() {
        let mut p = FramePacer::new(FrameConfig::with_frames(SyncTier::Sync2, 2));
        p.submit();
        assert!(!p.is_busy());
        p.submit();
        assert!(
            p.is_busy(),
            "two frames in flight with a limit of two is at capacity"
        );
    }

    #[test]
    fn completing_a_frame_frees_a_slot() {
        let mut p = FramePacer::new(FrameConfig::with_frames(SyncTier::Sync2, 2));
        p.submit();
        p.submit();
        assert!(p.is_busy());
        p.complete();
        assert!(!p.is_busy());
        assert_eq!(p.in_flight(), 1);
    }

    #[test]
    fn in_flight_never_goes_negative() {
        let mut p = FramePacer::new(FrameConfig::with_frames(SyncTier::Sync2, 2));
        p.complete();
        p.complete();
        assert_eq!(
            p.in_flight(),
            0,
            "more completions than submissions must not wrap"
        );
    }

    #[test]
    fn timeline_target_is_some_only_on_the_timeline_path() {
        let mut t = FramePacer::new(FrameConfig::for_tier(SyncTier::Sync2Timeline));
        t.submit();
        assert_eq!(t.timeline_target(), Some(1));

        let b = FramePacer::new(FrameConfig::for_tier(SyncTier::Legacy));
        assert_eq!(
            b.timeline_target(),
            None,
            "the binary path waits on a fence"
        );
    }

    #[test]
    fn timeline_target_tracks_submissions() {
        let mut p = FramePacer::new(FrameConfig::for_tier(SyncTier::Sync2Timeline));
        p.submit();
        p.submit();
        assert_eq!(p.timeline_target(), Some(2));
    }

    #[test]
    fn uses_timeline_reflects_the_config() {
        assert!(FramePacer::new(FrameConfig::for_tier(SyncTier::Sync2Timeline)).uses_timeline());
        assert!(!FramePacer::new(FrameConfig::for_tier(SyncTier::Sync2)).uses_timeline());
    }

    #[test]
    fn wait_slot_advances_with_completions() {
        let mut p = FramePacer::new(FrameConfig::with_frames(SyncTier::Sync2, 2));
        assert_eq!(p.wait_slot(), 0);
        p.complete();
        assert_eq!(p.wait_slot(), 1);
        p.complete();
        assert_eq!(p.wait_slot(), 0);
    }

    #[test]
    fn reset_clears_the_counters() {
        let mut p = FramePacer::new(FrameConfig::with_frames(SyncTier::Sync2, 2));
        p.submit();
        p.reset();
        assert_eq!(p.submitted, 0);
        assert_eq!(p.completed, 0);
        assert_eq!(p.in_flight(), 0);
    }

    #[test]
    fn deferred_queue_starts_empty() {
        let q = DeferredDestructionQueue::new();
        assert!(q.is_empty());
        assert_eq!(q.len(), 0);
    }

    #[test]
    fn fenced_objects_wait_for_their_fence() {
        let mut q = DeferredDestructionQueue::new();
        let fence = vk::Fence::from_raw(1);
        q.push_fenced(fence, 3, 100);
        assert_eq!(q.len(), 1);

        // Not yet signalled.
        assert!(q.collect_ready(0, |_, _| false).is_empty());
        assert_eq!(q.len(), 1);

        // Now signalled.
        let ready = q.collect_ready(0, |_, _| true);
        assert_eq!(ready, vec![100]);
        assert!(q.is_empty());
    }

    #[test]
    fn fenced_objects_only_release_at_their_own_value() {
        let mut q = DeferredDestructionQueue::new();
        let fence = vk::Fence::from_raw(1);
        q.push_fenced(fence, 5, 100);
        // The fence has reached 2, not 5.
        let ready = q.collect_ready(0, |_, value| 2 >= value);
        assert!(ready.is_empty());
    }

    #[test]
    fn timeline_objects_wait_for_their_value() {
        let mut q = DeferredDestructionQueue::new();
        q.push_timeline(4, 200);
        assert!(q.collect_ready(3, |_, _| true).is_empty());
        assert_eq!(q.collect_ready(4, |_, _| true), vec![200]);
        assert!(q.is_empty());
    }

    #[test]
    fn both_kinds_release_together() {
        let mut q = DeferredDestructionQueue::new();
        q.push_timeline(1, 10);
        q.push_fenced(vk::Fence::from_raw(1), 1, 20);
        let ready = q.collect_ready(1, |_, _| true);
        assert_eq!(ready.len(), 2);
        assert!(ready.contains(&10) && ready.contains(&20));
        assert!(q.is_empty());
    }

    #[test]
    fn clear_drops_everything_without_freeing() {
        let mut q = DeferredDestructionQueue::new();
        q.push_timeline(1, 10);
        q.push_fenced(vk::Fence::from_raw(1), 1, 20);
        q.clear();
        assert!(q.is_empty());
    }

    #[test]
    fn a_frame_cycle_keeps_the_pacer_in_step() {
        let config = FrameConfig::with_frames(SyncTier::Sync2Timeline, 2);
        let mut p = FramePacer::new(config);
        for i in 0..10u64 {
            let slot = p.submit();
            assert_eq!(slot, (i % 2) as usize, "frame {i} landed in the wrong slot");
            // The previous frame completes before this one is submitted.
            if i > 0 {
                p.complete();
            }
        }
        assert_eq!(p.submitted, 10);
        assert_eq!(p.completed, 9);
    }
}
