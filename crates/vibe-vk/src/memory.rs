//! GPU memory allocation and the buffer/image resources built on it.
//!
//! A full VMA is the eventual answer, but the part that actually decides
//! correctness is memory-type selection: pick the wrong type and a buffer is
//! either invisible to the GPU or cannot be mapped for upload. That decision
//! is a pure function of the heap layout and what the resource needs, so it is
//! here and tested; the `vkAllocateMemory` call is a thin wrapper over it.

use ash::vk;
use log::debug;

use crate::context::VkError;

/// What a resource needs from its memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct MemoryNeed {
    /// The CPU must be able to write the memory directly.
    pub host_visible: bool,
    /// The CPU must be able to read it back.
    ///
    /// Satisfied by `HOST_VISIBLE`; `HOST_CACHED` is preferred but not
    /// required, because a `HOST_VISIBLE | HOST_COHERENT` type can be read
    /// from, just less efficiently.
    pub host_readable: bool,
    /// The GPU writes it every frame, so it should be in device-local memory.
    pub device_local: bool,
}

impl MemoryNeed {
    /// Upload memory: CPU writes, GPU reads.
    pub fn upload() -> MemoryNeed {
        MemoryNeed {
            host_visible: true,
            host_readable: true,
            device_local: false,
        }
    }

    /// GPU-only storage, which is what vertex and index buffers want.
    pub fn device_local() -> MemoryNeed {
        MemoryNeed {
            device_local: true,
            ..Default::default()
        }
    }

    /// Memory the CPU reads and writes every frame, e.g. a uniform ring.
    pub fn host_cached() -> MemoryNeed {
        MemoryNeed {
            host_visible: true,
            host_readable: true,
            device_local: true,
        }
    }
}

/// The heap layout a device reported.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct HeapLayout {
    /// Property flags per memory type.
    pub property_flags: Vec<vk::MemoryPropertyFlags>,
    /// Which heap each memory type belongs to.
    pub heap_index: Vec<u32>,
    /// Bytes available in each heap.
    pub heap_sizes: Vec<u64>,
}

/// Choose a memory type index for a resource's needs.
///
/// Scores every compatible type and takes the best, so a device offering both
/// a `HOST_VISIBLE | HOST_CACHED` type and a `HOST_VISIBLE` type picks the
/// cached one. Ties go to the lowest index, which keeps the choice stable
/// frame to frame.
///
/// Returns `None` when no type satisfies the request, which is a real
/// possibility for a device-local need on a device with no such heap.
///
/// # Safety
///
/// The layout must be the one reported for a live physical device.
pub fn choose_memory_type(layout: &HeapLayout, need: MemoryNeed) -> Option<u32> {
    let mut best: Option<(i32, u32, i32)> = None;

    for (index, flags) in layout.property_flags.iter().enumerate() {
        if need.host_visible && !flags.contains(vk::MemoryPropertyFlags::HOST_VISIBLE) {
            continue;
        }
        if need.host_readable && !flags.contains(vk::MemoryPropertyFlags::HOST_VISIBLE) {
            continue;
        }
        if need.device_local && !flags.contains(vk::MemoryPropertyFlags::DEVICE_LOCAL) {
            continue;
        }

        // Score against what was *asked for*, not against an absolute
        // hierarchy. A pure device-local request must not drift to a
        // host-visible type just because that type is also device-local (true
        // on unified memory), and a host-cached request should not settle for a
        // merely coherent type.
        let mut score = 0i32;
        if need.device_local && flags.contains(vk::MemoryPropertyFlags::DEVICE_LOCAL) {
            score += 100;
        }
        if !need.device_local && !flags.contains(vk::MemoryPropertyFlags::DEVICE_LOCAL) {
            // For uploads, staying off device-local memory keeps the fast
            // heaps free for geometry.
            score += 20;
        }
        if need.host_visible || need.host_readable {
            if flags.contains(vk::MemoryPropertyFlags::HOST_CACHED) {
                score += 10;
            } else if flags.contains(vk::MemoryPropertyFlags::HOST_COHERENT) {
                score += 5;
            }
            if flags.contains(vk::MemoryPropertyFlags::HOST_VISIBLE) {
                score += 1;
            }
        }
        // Prefer the largest heap, then the lowest index, for stability.
        let heap = layout.heap_index.get(index).copied().unwrap_or(0);
        let size = layout.heap_sizes.get(heap as usize).copied().unwrap_or(0);
        // Negated so the max comparison breaks ties toward the lowest
        // index, which keeps the choice identical frame to frame.
        let key = (score, (size >> 20) as u32, -(index as i32));

        if best.is_none_or(|b| key > b) {
            best = Some(key);
        }
    }

    best.map(|(_, _, index)| (-index) as u32)
}

/// The usage flags a buffer needs, given what it will be bound as.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct BufferUsage {
    /// Bound as vertex data.
    pub vertex: bool,
    /// Bound as index data.
    pub index: bool,
    /// Bound as a uniform or storage buffer.
    pub uniform: bool,
    /// Copied into with a transfer command.
    pub transfer_src: bool,
    /// Copied out of with a transfer command.
    pub transfer_dst: bool,
}

impl BufferUsage {
    /// The Vulkan usage flags for this combination.
    pub fn flags(self) -> vk::BufferUsageFlags {
        let mut f = vk::BufferUsageFlags::empty();
        if self.vertex {
            f |= vk::BufferUsageFlags::VERTEX_BUFFER;
        }
        if self.index {
            f |= vk::BufferUsageFlags::INDEX_BUFFER;
        }
        if self.uniform {
            f |= vk::BufferUsageFlags::UNIFORM_BUFFER | vk::BufferUsageFlags::STORAGE_BUFFER;
        }
        if self.transfer_src {
            f |= vk::BufferUsageFlags::TRANSFER_SRC;
        }
        if self.transfer_dst {
            f |= vk::BufferUsageFlags::TRANSFER_DST;
        }
        f
    }

    /// True when the buffer needs no usage flags, which Vulkan rejects.
    pub fn is_empty(self) -> bool {
        self.flags().is_empty()
    }
}

/// A GPU buffer and the memory backing it.
pub struct GpuBuffer {
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    size: vk::DeviceSize,
    /// The address `map_memory` returned, kept so mapped reads and writes do
    /// not have to re-map.
    base_ptr: *mut u8,
    mapped: bool,
}

impl std::fmt::Debug for GpuBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuBuffer")
            .field("buffer", &self.buffer)
            .field("size", &self.size)
            .field("mapped", &self.mapped)
            .finish()
    }
}

impl GpuBuffer {
    /// Create a buffer and allocate memory for it.
    ///
    /// # Safety
    ///
    /// `device` must be a live logical device and `layout` its physical
    /// device's heap layout.
    pub unsafe fn create(
        device: &ash::Device,
        layout: &HeapLayout,
        size: vk::DeviceSize,
        usage: BufferUsage,
        need: MemoryNeed,
    ) -> Result<GpuBuffer, VkError> {
        if size == 0 {
            return Err(VkError::Swapchain(
                "buffer size must be non-zero".to_string(),
            ));
        }
        let flags = usage.flags();
        if flags.is_empty() {
            return Err(VkError::Swapchain(
                "buffer needs at least one usage flag".to_string(),
            ));
        }

        let buffer_info = vk::BufferCreateInfo {
            size,
            usage: flags,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            ..Default::default()
        };

        // Every step below can fail, and each failure must undo the ones
        // before it, so the cleanup is written out rather than left to a drop
        // guard that would need the device handle.
        unsafe {
            let buffer = device.create_buffer(&buffer_info, None)?;
            let requirements = device.get_buffer_memory_requirements(buffer);

            let Some(memory_type) = choose_memory_type(layout, need) else {
                device.destroy_buffer(buffer, None);
                return Err(VkError::Swapchain(format!(
                    "no memory type satisfies {need:?}"
                )));
            };

            let alloc_info = vk::MemoryAllocateInfo {
                allocation_size: requirements.size,
                memory_type_index: memory_type,
                ..Default::default()
            };
            let memory = match device.allocate_memory(&alloc_info, None) {
                Ok(m) => m,
                Err(e) => {
                    device.destroy_buffer(buffer, None);
                    return Err(VkError::from(e));
                }
            };

            if let Err(e) = device.bind_buffer_memory(buffer, memory, 0) {
                device.destroy_buffer(buffer, None);
                device.free_memory(memory, None);
                return Err(VkError::from(e));
            }

            debug!("buffer created: {size} bytes, memory type {memory_type}");
            Ok(GpuBuffer {
                buffer,
                memory,
                size,
                base_ptr: std::ptr::null_mut(),
                mapped: false,
            })
        }
    }

    /// The buffer handle.
    pub fn buffer(&self) -> vk::Buffer {
        self.buffer
    }

    /// The memory handle.
    pub fn memory(&self) -> vk::DeviceMemory {
        self.memory
    }

    /// The buffer's size in bytes.
    pub fn size(&self) -> vk::DeviceSize {
        self.size
    }

    /// True when the memory is currently mapped.
    pub fn is_mapped(&self) -> bool {
        self.mapped
    }

    /// Map the memory for CPU access.
    ///
    /// # Safety
    ///
    /// The memory must have been allocated with `HOST_VISIBLE`, and nothing
    /// may be using the buffer concurrently.
    pub unsafe fn map(&mut self, device: &ash::Device) -> Result<(), VkError> {
        if self.mapped {
            return Ok(());
        }
        let ptr = unsafe {
            device.map_memory(self.memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty())
        }?;
        if ptr.is_null() {
            return Err(VkError::Swapchain("map_memory returned null".to_string()));
        }
        self.base_ptr = ptr.cast();
        self.mapped = true;
        Ok(())
    }

    /// Unmap the memory.
    pub unsafe fn unmap(&mut self, device: &ash::Device) {
        if self.mapped {
            unsafe {
                device.unmap_memory(self.memory);
            }
            self.base_ptr = std::ptr::null_mut();
            self.mapped = false;
        }
    }

    /// Write bytes into a mapped buffer.
    ///
    /// # Safety
    ///
    /// The buffer must be mapped, and `data` must fit within `size`.
    pub unsafe fn write_mapped(&self, offset: u64, data: &[u8]) -> Result<(), VkError> {
        if !self.mapped {
            return Err(VkError::Swapchain("buffer is not mapped".to_string()));
        }
        if offset + data.len() as u64 > self.size {
            return Err(VkError::Swapchain(format!(
                "write of {} bytes at {offset} overruns a {}-byte buffer",
                data.len(),
                self.size
            )));
        }
        let ptr = self.mapped_ptr(offset);
        unsafe {
            std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, data.len());
        }
        Ok(())
    }

    /// Read bytes out of a mapped buffer.
    ///
    /// # Safety
    ///
    /// The buffer must be mapped, and the range must lie within it.
    pub unsafe fn read_mapped(&self, offset: u64, len: usize) -> Result<Vec<u8>, VkError> {
        if !self.mapped {
            return Err(VkError::Swapchain("buffer is not mapped".to_string()));
        }
        if offset + len as u64 > self.size {
            return Err(VkError::Swapchain("read overruns the buffer".to_string()));
        }
        let ptr = self.mapped_ptr(offset);
        Ok(unsafe { std::slice::from_raw_parts(ptr, len) }.to_vec())
    }

    /// The address of a mapped range, recorded at map time.
    fn mapped_ptr(&self, offset: u64) -> *mut u8 {
        if self.base_ptr.is_null() {
            std::ptr::null_mut()
        } else {
            unsafe { self.base_ptr.add(offset as usize) }
        }
    }

    /// Destroy the buffer and free its memory.
    ///
    /// # Safety
    ///
    /// The buffer must not be in use by the GPU, and must be unmapped.
    pub unsafe fn destroy(&mut self, device: &ash::Device) {
        unsafe {
            self.unmap(device);
        }
        unsafe {
            device.destroy_buffer(self.buffer, None);
            device.free_memory(self.memory, None);
        }
        self.buffer = vk::Buffer::null();
        self.memory = vk::DeviceMemory::null();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A typical integrated-GPU layout: one device-local heap with a cached
    /// host-visible type and a plain host-visible type on the same heap.
    fn integrated_layout() -> HeapLayout {
        HeapLayout {
            property_flags: vec![
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_CACHED,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
                vk::MemoryPropertyFlags::DEVICE_LOCAL | vk::MemoryPropertyFlags::HOST_VISIBLE,
            ],
            heap_index: vec![0, 0, 0, 0],
            heap_sizes: vec![8 << 30],
        }
    }

    /// A typical discrete layout with a small host-visible heap.
    fn discrete_layout() -> HeapLayout {
        HeapLayout {
            property_flags: vec![
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
                vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_CACHED,
            ],
            heap_index: vec![0, 1],
            heap_sizes: vec![8 << 30, 512 << 20],
        }
    }

    #[test]
    fn device_local_resources_prefer_device_local_memory() {
        let t = choose_memory_type(&integrated_layout(), MemoryNeed::device_local()).unwrap();
        assert_eq!(t, 0, "type 0 is the only pure device-local type");
    }

    #[test]
    fn uploads_prefer_a_cached_host_visible_type() {
        let t = choose_memory_type(&integrated_layout(), MemoryNeed::upload()).unwrap();
        assert_eq!(
            t, 1,
            "HOST_VISIBLE | HOST_CACHED beats HOST_VISIBLE | HOST_COHERENT"
        );
    }

    #[test]
    fn a_coherent_only_layout_still_yields_an_upload_type() {
        let layout = HeapLayout {
            property_flags: vec![vk::MemoryPropertyFlags::HOST_VISIBLE],
            heap_index: vec![0],
            heap_sizes: vec![1 << 30],
        };
        assert!(choose_memory_type(&layout, MemoryNeed::upload()).is_some());
    }

    #[test]
    fn a_layout_with_no_host_visible_type_cannot_host_upload() {
        let layout = HeapLayout {
            property_flags: vec![vk::MemoryPropertyFlags::DEVICE_LOCAL],
            heap_index: vec![0],
            heap_sizes: vec![1 << 30],
        };
        assert!(choose_memory_type(&layout, MemoryNeed::upload()).is_none());
    }

    #[test]
    fn a_layout_with_no_device_local_type_cannot_have_device_local_memory() {
        let layout = HeapLayout {
            property_flags: vec![vk::MemoryPropertyFlags::HOST_VISIBLE],
            heap_index: vec![0],
            heap_sizes: vec![1 << 30],
        };
        assert!(choose_memory_type(&layout, MemoryNeed::device_local()).is_none());
    }

    #[test]
    fn a_cached_device_local_type_wins_for_host_cached() {
        let t = choose_memory_type(&integrated_layout(), MemoryNeed::host_cached()).unwrap();
        assert_eq!(t, 3, "type 3 is DEVICE_LOCAL | HOST_VISIBLE");
    }

    #[test]
    fn the_choice_is_stable_across_calls() {
        let layout = integrated_layout();
        let first = choose_memory_type(&layout, MemoryNeed::upload());
        for _ in 0..10 {
            assert_eq!(choose_memory_type(&layout, MemoryNeed::upload()), first);
        }
    }

    #[test]
    fn discrete_layout_picks_the_small_heap_for_uploads() {
        let layout = discrete_layout();
        let t = choose_memory_type(&layout, MemoryNeed::upload()).unwrap();
        assert_eq!(
            layout.heap_index[t as usize], 1,
            "uploads belong on the host heap"
        );
    }

    #[test]
    fn an_empty_layout_yields_nothing() {
        let layout = HeapLayout::default();
        assert!(choose_memory_type(&layout, MemoryNeed::device_local()).is_none());
        assert!(choose_memory_type(&layout, MemoryNeed::upload()).is_none());
    }

    #[test]
    fn vertex_and_index_usage_produce_flags() {
        let f = BufferUsage {
            vertex: true,
            index: true,
            ..Default::default()
        }
        .flags();
        assert!(f.contains(vk::BufferUsageFlags::VERTEX_BUFFER));
        assert!(f.contains(vk::BufferUsageFlags::INDEX_BUFFER));
    }

    #[test]
    fn uniform_usage_covers_both_uniform_and_storage() {
        let f = BufferUsage {
            uniform: true,
            ..Default::default()
        }
        .flags();
        assert!(f.contains(vk::BufferUsageFlags::UNIFORM_BUFFER));
        assert!(f.contains(vk::BufferUsageFlags::STORAGE_BUFFER));
    }

    #[test]
    fn transfer_usage_produces_transfer_flags() {
        let f = BufferUsage {
            transfer_src: true,
            transfer_dst: true,
            ..Default::default()
        }
        .flags();
        assert!(f.contains(vk::BufferUsageFlags::TRANSFER_SRC));
        assert!(f.contains(vk::BufferUsageFlags::TRANSFER_DST));
    }

    #[test]
    fn usage_with_nothing_set_is_rejected() {
        // Vulkan rejects a buffer with no usage flags, so this must be caught
        // before the driver call.
        let u = BufferUsage::default();
        assert!(u.is_empty());
        assert!(u.flags().is_empty());
    }

    #[test]
    fn needs_cover_the_three_cases() {
        assert!(MemoryNeed::upload().host_visible);
        assert!(!MemoryNeed::device_local().host_visible);
        assert!(MemoryNeed::host_cached().device_local);
    }
}
