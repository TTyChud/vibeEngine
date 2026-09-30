//! Command buffer pools and the per-frame sync objects.
//!
//! One command buffer per frame in flight, each with its own fence, so a frame
//! can be recorded while an earlier one is still on the GPU. The bookkeeping
//! for which slots are free is pure logic and is tested without a device.

use std::sync::Arc;

use ash::vk;
use log::debug;

use crate::error::FrameError;

/// How many frames may be in flight, and which sync objects that needs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FrameConfig {
    /// Frames in flight, at least 1.
    pub frames_in_flight: usize,
    /// Whether to use one timeline semaphore rather than per-frame binaries.
    pub use_timeline: bool,
}

impl FrameConfig {
    /// A configuration for a device on the given sync tier.
    pub fn for_tier(tier: vibe_vk::SyncTier) -> FrameConfig {
        FrameConfig {
            frames_in_flight: vibe_vk::default_frames_in_flight(),
            use_timeline: tier.supports_timeline(),
        }
    }

    /// A configuration with an explicit frame count.
    ///
    /// # Panics
    ///
    /// Panics if `frames_in_flight` is zero, which would deadlock the wait.
    pub fn with_frames(tier: vibe_vk::SyncTier, frames_in_flight: usize) -> FrameConfig {
        assert!(frames_in_flight > 0, "frames_in_flight must be at least 1");
        FrameConfig {
            frames_in_flight,
            ..FrameConfig::for_tier(tier)
        }
    }

    /// Number of command buffers to allocate.
    pub fn command_buffers(&self) -> usize {
        self.frames_in_flight
    }

    /// Number of WSI semaphores: two per frame on every tier.
    ///
    /// `vkAcquireNextImageKHR` and `vkQueuePresentKHR` both require binary
    /// semaphores with no value field, so the timeline path cannot replace them.
    /// Two per frame rather than one because a submit may not use the same
    /// semaphore as both its wait and its signal: the frame waits on the one
    /// acquire signalled and signals the one present waits on.
    pub fn wsi_semaphores(&self) -> usize {
        self.frames_in_flight * 2
    }

    /// Number of fences: one per frame on both paths, so a frame can wait on
    /// its own submission.
    pub fn fences(&self) -> usize {
        self.frames_in_flight
    }
}

/// A command pool and the buffers allocated from it.
///
/// The pool is reset as a whole rather than individual buffers: `vkResetCommandPool`
/// is one call and a single barrier, where per-buffer resets would need one each.
pub struct CommandPool {
    pool: vk::CommandPool,
    buffers: Vec<vk::CommandBuffer>,
    device: vk::Device,
}

impl std::fmt::Debug for CommandPool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CommandPool")
            .field("buffers", &self.buffers.len())
            .field("pool", &self.pool)
            .finish()
    }
}

impl CommandPool {
    /// Create a pool and allocate one buffer per frame.
    ///
    /// # Safety
    ///
    /// `device` must be a live logical device, and `queue_family` must be a
    /// family it has a queue on.
    pub unsafe fn new(
        device: &ash::Device,
        queue_family: u32,
        frames_in_flight: usize,
    ) -> Result<CommandPool, FrameError> {
        unsafe {
            let info = vk::CommandPoolCreateInfo {
                // One pool per frame avoids an implicit barrier between frames: each
                // frame's writes are independent of the others'.
                queue_family_index: queue_family,
                flags: vk::CommandPoolCreateFlags::TRANSIENT
                    | vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
                ..Default::default()
            };
            let pool = device.create_command_pool(&info, None)?;

            let alloc = vk::CommandBufferAllocateInfo {
                command_pool: pool,
                level: vk::CommandBufferLevel::PRIMARY,
                command_buffer_count: frames_in_flight as u32,
                ..Default::default()
            };
            let buffers = match device.allocate_command_buffers(&alloc) {
                Ok(b) => b,
                Err(e) => {
                    device.destroy_command_pool(pool, None);
                    return Err(FrameError::Vk(e));
                }
            };

            debug!("command pool with {} buffer(s)", buffers.len());
            Ok(CommandPool {
                pool,
                buffers,
                device: device.handle(),
            })
        }
    }

    /// The pool handle.
    pub fn handle(&self) -> vk::CommandPool {
        self.pool
    }

    /// The command buffers, indexed by frame slot.
    pub fn buffers(&self) -> &[vk::CommandBuffer] {
        &self.buffers
    }

    /// The device this pool belongs to, checked at teardown.
    pub fn device(&self) -> vk::Device {
        self.device
    }

    /// The command buffer for a frame slot.
    pub fn buffer(&self, slot: usize) -> Option<vk::CommandBuffer> {
        self.buffers.get(slot).copied()
    }

    /// How many buffers the pool holds.
    pub fn len(&self) -> usize {
        self.buffers.len()
    }

    /// True when the pool holds no buffers.
    pub fn is_empty(&self) -> bool {
        self.buffers.is_empty()
    }

    /// Begin recording a frame's command buffer.
    ///
    /// # Safety
    ///
    /// The slot must not be recording, and its GPU work must have finished.
    pub unsafe fn begin(
        &self,
        device: &ash::Device,
        slot: usize,
    ) -> Result<vk::CommandBuffer, FrameError> {
        let buffer = self
            .buffer(slot)
            .ok_or(FrameError::NotRecording(slot as u32))?;
        let info = vk::CommandBufferBeginInfo {
            // One-time-use is the whole point: each buffer is recorded once per
            // frame and never re-recorded, so the driver can preallocate.
            flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
            ..Default::default()
        };
        unsafe {
            device.begin_command_buffer(buffer, &info)?;
        }
        Ok(buffer)
    }

    /// Finish recording a frame's command buffer.
    ///
    /// # Safety
    ///
    /// The buffer must be recording.
    pub unsafe fn end(
        &self,
        device: &ash::Device,
        slot: usize,
    ) -> Result<vk::CommandBuffer, FrameError> {
        let buffer = self
            .buffer(slot)
            .ok_or(FrameError::NotRecording(slot as u32))?;
        unsafe {
            device.end_command_buffer(buffer)?;
        }
        Ok(buffer)
    }

    /// Reset the pool so its buffers can be recorded again.
    ///
    /// # Safety
    ///
    /// Every buffer must be finished with by the GPU.
    pub unsafe fn reset(&self, device: &ash::Device) -> Result<(), FrameError> {
        unsafe { device.reset_command_pool(self.pool, vk::CommandPoolResetFlags::empty()) }?;
        Ok(())
    }

    /// Destroy the pool and its buffers.
    ///
    /// # Safety
    ///
    /// The device must still be alive and no work may be in flight.
    pub unsafe fn destroy(&mut self, device: &ash::Device) {
        unsafe {
            device.destroy_command_pool(self.pool, None);
        }
        self.pool = vk::CommandPool::null();
        self.buffers.clear();
    }
}

/// The sync objects every frame needs.
pub struct FrameResources {
    /// One fence per frame slot.
    pub fences: Vec<vk::Fence>,
    /// Per-frame WSI semaphores, two per frame: acquire and present.
    ///
    /// Present on every tier, unlike the timeline semaphore, because
    /// `vkAcquireNextImageKHR` and `vkQueuePresentKHR` only accept binary ones.
    pub wsi_semaphores: Vec<vk::Semaphore>,
    /// The single timeline semaphore, when the tier supports one.
    pub timeline: Option<vk::Semaphore>,
    /// The value the last submitted frame will signal.
    pub submitted_value: u64,
    /// The number of frames submitted so far.
    pub submitted: u64,
    /// The number of frames the GPU has finished.
    pub completed: u64,
    config: FrameConfig,
}

impl std::fmt::Debug for FrameResources {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FrameResources")
            .field("frames_in_flight", &self.config.frames_in_flight)
            .field("timeline", &self.timeline.is_some())
            .field("submitted", &self.submitted)
            .field("completed", &self.completed)
            .finish()
    }
}

impl FrameResources {
    /// Create the fences and semaphores for a configuration.
    ///
    /// # Safety
    ///
    /// `device` must be a live logical device.
    pub unsafe fn new(
        device: &ash::Device,
        config: FrameConfig,
    ) -> Result<FrameResources, FrameError> {
        unsafe {
            let fence_info = vk::FenceCreateInfo {
                flags: vk::FenceCreateFlags::empty(),
                ..Default::default()
            };
            let mut fences = Vec::with_capacity(config.fences());
            for _ in 0..config.fences() {
                match device.create_fence(&fence_info, None) {
                    Ok(f) => fences.push(f),
                    Err(e) => {
                        for f in fences {
                            device.destroy_fence(f, None);
                        }
                        return Err(FrameError::Vk(e));
                    }
                }
            }

            let mut wsi_semaphores = Vec::with_capacity(config.wsi_semaphores());
            let timeline = if config.use_timeline {
                // A timeline semaphore is a binary semaphore with a pNext carrying
                // the initial value; ash exposes one create call for both.
                let initial = vk::SemaphoreTypeCreateInfo {
                    semaphore_type: vk::SemaphoreType::TIMELINE,
                    initial_value: 0,
                    ..Default::default()
                };
                let info = vk::SemaphoreCreateInfo {
                    p_next: &initial as *const _ as *const std::ffi::c_void,
                    ..Default::default()
                };
                match device.create_semaphore(&info, None) {
                    Ok(s) => Some(s),
                    Err(e) => {
                        for f in fences {
                            device.destroy_fence(f, None);
                        }
                        return Err(FrameError::Vk(e));
                    }
                }
            } else {
                None
            };

            // Binary semaphores are needed on every tier: the WSI calls take
            // no value and have no timeline equivalent.
            {
                let info = vk::SemaphoreCreateInfo::default();
                for _ in 0..config.wsi_semaphores() {
                    match device.create_semaphore(&info, None) {
                        Ok(s) => wsi_semaphores.push(s),
                        Err(e) => {
                            for f in fences {
                                device.destroy_fence(f, None);
                            }
                            for s in wsi_semaphores {
                                device.destroy_semaphore(s, None);
                            }
                            return Err(FrameError::Vk(e));
                        }
                    }
                }
            }

            Ok(FrameResources {
                fences,
                wsi_semaphores,
                timeline,
                submitted_value: 0,
                submitted: 0,
                completed: 0,
                config,
            })
        }
    }

    /// The configuration these resources were built for.
    pub fn config(&self) -> FrameConfig {
        self.config
    }

    /// How many frames may be in flight.
    pub fn frames_in_flight(&self) -> usize {
        self.config.frames_in_flight
    }

    /// The slot for the next frame, and the value it will signal.
    ///
    /// # Errors
    ///
    /// Returns [`FrameError::FramesBusy`] when the GPU is still `frames_in_flight`
    /// behind, which is the point at which the CPU must stop recording.
    pub fn next_slot(&mut self) -> Result<(usize, u64), FrameError> {
        if self.in_flight() >= self.config.frames_in_flight as u64 {
            return Err(FrameError::FramesBusy(self.config.frames_in_flight));
        }
        let slot = (self.submitted % self.config.frames_in_flight as u64) as usize;
        let value = self.submitted + 1;
        self.submitted += 1;
        self.submitted_value = value;
        Ok((slot, value))
    }

    /// Record that a frame was submitted.
    pub fn on_submit(&mut self) {
        // submitted_value was already advanced by next_slot.
    }

    /// Record that a frame's GPU work finished.
    pub fn on_complete(&mut self) {
        self.completed += 1;
    }

    /// How many frames the CPU is ahead of the GPU.
    pub fn in_flight(&self) -> u64 {
        self.submitted.saturating_sub(self.completed)
    }

    /// The fence for a slot.
    pub fn fence(&self, slot: usize) -> Option<vk::Fence> {
        self.fences.get(slot).copied()
    }

    /// The semaphore `vkAcquireNextImageKHR` signals for a slot.
    pub fn acquire_semaphore(&self, slot: usize) -> Option<vk::Semaphore> {
        self.wsi_semaphores.get(slot * 2).copied()
    }

    /// The semaphore the frame signals for `vkQueuePresentKHR` to wait on.
    pub fn present_semaphore(&self, slot: usize) -> Option<vk::Semaphore> {
        self.wsi_semaphores.get(slot * 2 + 1).copied()
    }

    /// The timeline semaphore, on the timeline path.
    pub fn timeline_semaphore(&self) -> Option<vk::Semaphore> {
        self.timeline
    }

    /// Wait for a frame's fence.
    ///
    /// # Safety
    ///
    /// The fence must belong to a live device.
    pub unsafe fn wait(&self, device: &ash::Device, slot: usize) -> Result<(), FrameError> {
        let fence = self
            .fence(slot)
            .ok_or(FrameError::NotRecording(slot as u32))?;
        unsafe {
            device.wait_for_fences(&[fence], true, u64::MAX)?;
        }
        Ok(())
    }

    /// Destroy every sync object.
    ///
    /// # Safety
    ///
    /// The device must still be alive and no work may be in flight.
    pub unsafe fn destroy(&mut self, device: &ash::Device) {
        unsafe {
            for f in self.fences.drain(..) {
                device.destroy_fence(f, None);
            }
            for s in self.wsi_semaphores.drain(..) {
                device.destroy_semaphore(s, None);
            }
            if let Some(t) = self.timeline.take() {
                device.destroy_semaphore(t, None);
            }
        }
    }
}

/// A frame's resources, shared between the renderer and the frame loop.
pub type SharedFrameResources = Arc<FrameResources>;

#[cfg(test)]
mod tests {
    use super::*;
    use ash::vk::Handle as _;
    use vibe_vk::SyncTier;

    #[test]
    fn a_timeline_config_still_creates_wsi_semaphores() {
        // The timeline semaphore cannot be passed to vkAcquireNextImageKHR,
        // which requires a binary one, so every tier needs these.
        let c = FrameConfig::for_tier(SyncTier::Sync2Timeline);
        assert!(c.use_timeline);
        assert_eq!(c.wsi_semaphores(), c.frames_in_flight * 2);
    }

    #[test]
    fn a_binary_config_uses_two_per_frame() {
        let c = FrameConfig::for_tier(SyncTier::Legacy);
        assert!(!c.use_timeline);
        assert_eq!(c.wsi_semaphores(), c.frames_in_flight * 2);
    }

    #[test]
    fn every_tier_gets_a_fence_per_frame() {
        for tier in [SyncTier::Legacy, SyncTier::Sync2, SyncTier::Sync2Timeline] {
            let c = FrameConfig::for_tier(tier);
            assert_eq!(c.fences(), c.frames_in_flight, "{tier}");
        }
    }

    #[test]
    fn command_buffers_match_the_frame_count() {
        assert_eq!(
            FrameConfig::with_frames(SyncTier::Sync2, 3).command_buffers(),
            3
        );
    }

    #[test]
    #[should_panic(expected = "at least 1")]
    fn zero_frames_panics() {
        FrameConfig::with_frames(SyncTier::Sync2, 0);
    }

    /// A resource set with no Vulkan objects, for testing the slot logic.
    fn bookkeeping(config: FrameConfig) -> FrameResources {
        FrameResources {
            fences: vec![vk::Fence::null(); config.fences()],
            wsi_semaphores: (0..config.wsi_semaphores())
                // Distinct handles: a test that checks the pairs differ cannot
                // tell null from null.
                .map(|i| vk::Semaphore::from_raw(i as u64 + 1))
                .collect(),
            timeline: config.use_timeline.then_some(vk::Semaphore::null()),
            submitted_value: 0,
            submitted: 0,
            completed: 0,
            config,
        }
    }

    #[test]
    fn slots_advance_and_wrap() {
        let mut r = bookkeeping(FrameConfig::with_frames(SyncTier::Sync2, 2));
        assert_eq!(r.next_slot().unwrap().0, 0);
        assert_eq!(r.next_slot().unwrap().0, 1);
        r.on_complete();
        assert_eq!(r.next_slot().unwrap().0, 0, "the freed slot is reused");
    }

    #[test]
    fn a_frame_signals_the_next_value() {
        let mut r = bookkeeping(FrameConfig::with_frames(SyncTier::Sync2Timeline, 2));
        assert_eq!(r.next_slot().unwrap().1, 1);
        assert_eq!(r.next_slot().unwrap().1, 2);
    }

    #[test]
    fn the_counters_stay_balanced() {
        let mut r = bookkeeping(FrameConfig::with_frames(SyncTier::Sync2, 2));
        r.next_slot().unwrap();
        assert_eq!(r.in_flight(), 1);
        r.next_slot().unwrap();
        assert_eq!(r.in_flight(), 2);
        r.on_complete();
        assert_eq!(r.in_flight(), 1);
    }

    #[test]
    fn running_too_far_ahead_is_refused() {
        let mut r = bookkeeping(FrameConfig::with_frames(SyncTier::Sync2, 2));
        r.next_slot().unwrap();
        r.next_slot().unwrap();
        let err = r.next_slot().unwrap_err();
        assert!(matches!(err, FrameError::FramesBusy(2)), "{err:?}");
    }

    #[test]
    fn completing_a_frame_makes_room_again() {
        let mut r = bookkeeping(FrameConfig::with_frames(SyncTier::Sync2, 1));
        r.next_slot().unwrap();
        assert!(r.next_slot().is_err());
        r.on_complete();
        assert!(r.next_slot().is_ok());
    }

    #[test]
    fn in_flight_never_goes_negative() {
        let mut r = bookkeeping(FrameConfig::with_frames(SyncTier::Sync2, 2));
        r.on_complete();
        r.on_complete();
        assert_eq!(r.in_flight(), 0);
    }

    #[test]
    fn a_three_frame_config_cycles_three_slots() {
        let mut r = bookkeeping(FrameConfig::with_frames(SyncTier::Sync2, 3));
        for _ in 0..3 {
            r.next_slot().unwrap();
        }
        let mut seen: Vec<usize> = Vec::new();
        for _ in 0..3 {
            r.on_complete();
            seen.push(r.next_slot().unwrap().0);
        }
        seen.sort();
        assert_eq!(seen, vec![0, 1, 2]);
    }

    #[test]
    fn the_timeline_config_reports_its_semaphore() {
        let r = bookkeeping(FrameConfig::for_tier(SyncTier::Sync2Timeline));
        assert!(r.timeline_semaphore().is_some());
    }

    #[test]
    fn a_binary_config_has_no_timeline_semaphore() {
        let r = bookkeeping(FrameConfig::for_tier(SyncTier::Legacy));
        assert!(r.timeline_semaphore().is_none());
        assert!(r.acquire_semaphore(0).is_some());
    }

    #[test]
    fn acquire_and_present_use_different_semaphores() {
        // A submit may not wait on and signal the same semaphore.
        let r = bookkeeping(FrameConfig::with_frames(SyncTier::Sync2Timeline, 2));
        for slot in 0..2 {
            assert_ne!(
                r.acquire_semaphore(slot),
                r.present_semaphore(slot),
                "slot {slot}"
            );
        }
    }

    #[test]
    fn every_slot_has_its_own_pair() {
        let r = bookkeeping(FrameConfig::with_frames(SyncTier::Sync2Timeline, 2));
        let mut seen = std::collections::HashSet::new();
        for slot in 0..2 {
            seen.insert(r.acquire_semaphore(slot));
            seen.insert(r.present_semaphore(slot));
        }
        assert_eq!(seen.len(), 4, "two frames need four distinct semaphores");
    }

    #[test]
    fn a_slot_past_the_frame_count_has_no_pair() {
        let r = bookkeeping(FrameConfig::with_frames(SyncTier::Sync2Timeline, 2));
        assert!(r.acquire_semaphore(2).is_none());
        assert!(r.present_semaphore(2).is_none());
    }

    #[test]
    fn a_timeline_config_still_answers_the_wsi_calls() {
        let r = bookkeeping(FrameConfig::for_tier(SyncTier::Sync2Timeline));
        assert!(r.acquire_semaphore(0).is_some());
        assert!(r.present_semaphore(0).is_some());
    }

    #[test]
    fn an_out_of_range_slot_has_no_fence() {
        let r = bookkeeping(FrameConfig::with_frames(SyncTier::Sync2, 2));
        assert!(r.fence(0).is_some());
        assert!(r.fence(5).is_none());
    }

    #[test]
    fn the_config_is_reported_back() {
        let c = FrameConfig::with_frames(SyncTier::Sync2, 3);
        let r = bookkeeping(c);
        assert_eq!(r.config(), c);
        assert_eq!(r.frames_in_flight(), 3);
    }

    #[test]
    fn the_debug_form_names_the_sync_mode() {
        let r = bookkeeping(FrameConfig::for_tier(SyncTier::Sync2Timeline));
        let s = format!("{r:?}");
        assert!(s.contains("timeline: true"), "{s}");
    }

    #[test]
    fn a_ten_frame_run_keeps_the_counters_exact() {
        let mut r = bookkeeping(FrameConfig::with_frames(SyncTier::Sync2Timeline, 3));
        for i in 0..10u64 {
            let (slot, value) = r.next_slot().unwrap();
            assert_eq!(value, i + 1);
            assert_eq!(slot, (i % 3) as usize);
            r.on_complete();
        }
        assert_eq!(r.submitted, 10);
        assert_eq!(r.completed, 10);
        assert_eq!(r.in_flight(), 0);
    }
}
