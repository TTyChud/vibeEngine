//! The resource registry: stable handles with generation counters.

use vibe_rhi::ResourceHandle;

/// What a resource is, which decides how the graph tracks its lifetime.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum ResourceDesc {
    /// A buffer the graph does not own the memory of.
    Buffer {
        /// Size in bytes.
        size: u64,
    },
    /// A texture the graph does not own the memory of.
    Texture {
        /// Width in texels.
        width: u32,
        /// Height in texels.
        height: u32,
        /// Opaque backend format id.
        format: u32,
    },
    /// A transient resource the graph allocates and frees each frame.
    ///
    /// Transient resources are the point of a frame graph: they are cheap to
    /// alias, so the allocator can reuse the same memory for a colour target in
    /// one pass and a depth target in the next, as long as their passes do not
    /// overlap.
    Transient {
        /// Size in bytes.
        size: u64,
        /// Opaque backend format id.
        format: u32,
    },
}

impl ResourceDesc {
    /// Size in bytes, for aliasing and budgeting.
    pub fn size(&self) -> u64 {
        match self {
            ResourceDesc::Buffer { size } | ResourceDesc::Transient { size, .. } => *size,
            ResourceDesc::Texture {
                width,
                height,
                format: _,
            } => {
                // 4 bytes per texel is the common case; the real size comes
                // from the backend at allocation time.
                (*width as u64) * (*height as u64) * 4
            }
        }
    }

    /// True when the graph may alias this resource's memory.
    pub fn is_transient(&self) -> bool {
        matches!(self, ResourceDesc::Transient { .. })
    }

    /// A short label for dumps and error messages.
    pub fn label(&self) -> &'static str {
        match self {
            ResourceDesc::Buffer { .. } => "buffer",
            ResourceDesc::Texture { .. } => "texture",
            ResourceDesc::Transient { .. } => "transient",
        }
    }
}

/// A resource slot, kept even after the resource is freed.
#[derive(Debug, Clone)]
struct Slot {
    generation: u32,
    desc: Option<ResourceDesc>,
    /// Monotonic id bumped on every change, so a consumer can tell whether its
    /// cached view of the resource is stale without re-querying.
    version: u64,
    name: String,
}

/// Allocates and recycles resource slots.
///
/// The generation counter is what makes a handle safe: freeing slot 3 bumps its
/// generation, so a handle held across the free resolves to nothing rather than
/// to whatever now occupies slot 3.
#[derive(Debug, Default)]
pub struct ResourceRegistry {
    slots: Vec<Slot>,
    free: Vec<u32>,
    live: usize,
    /// Indices created through `import`, which the executor must not free.
    imported: Vec<u32>,
}

impl ResourceRegistry {
    /// An empty registry.
    pub fn new() -> ResourceRegistry {
        ResourceRegistry {
            slots: Vec::new(),
            free: Vec::new(),
            live: 0,
            imported: Vec::new(),
        }
    }

    /// Create a resource and return its handle.
    pub fn create(&mut self, desc: ResourceDesc, name: impl Into<String>) -> ResourceHandle {
        let index = match self.free.pop() {
            Some(i) => {
                self.slots[i as usize].desc = Some(desc);
                self.slots[i as usize].name = name.into();
                self.slots[i as usize].version += 1;
                i
            }
            None => {
                self.slots.push(Slot {
                    generation: 0,
                    desc: Some(desc),
                    version: 1,
                    name: name.into(),
                });
                (self.slots.len() - 1) as u32
            }
        };
        self.live += 1;
        ResourceHandle {
            index,
            generation: self.slots[index as usize].generation,
        }
    }

    /// Import a resource the graph does not own.
    ///
    /// Imported resources are never freed by the graph and are never aliased,
    /// which is what a swapchain image or an externally managed buffer needs.
    pub fn import(&mut self, desc: ResourceDesc, name: impl Into<String>) -> ResourceHandle {
        let handle = self.create(desc, name);
        self.imported.push(handle.index);
        handle
    }

    /// True when `handle` was produced by [`ResourceRegistry::import`].
    pub fn is_imported(&self, handle: ResourceHandle) -> bool {
        self.imported.contains(&handle.index)
    }

    /// Free a resource, recycling its slot.
    ///
    /// Returns the description that was freed, so the executor can destroy the
    /// backend object.
    pub fn free(&mut self, handle: ResourceHandle) -> Option<ResourceDesc> {
        if !self.is_alive(handle) {
            return None;
        }
        let slot = &mut self.slots[handle.index as usize];
        let desc = slot.desc.take();
        slot.generation = slot.generation.wrapping_add(1);
        slot.version += 1;
        slot.name.clear();
        self.free.push(handle.index);
        self.live -= 1;
        desc
    }

    /// True when `handle` refers to a live resource with a matching generation.
    pub fn is_alive(&self, handle: ResourceHandle) -> bool {
        handle.index < self.slots.len() as u32
            && self.slots[handle.index as usize].generation == handle.generation
            && self.slots[handle.index as usize].desc.is_some()
    }

    /// The description behind a live handle.
    pub fn desc(&self, handle: ResourceHandle) -> Option<&ResourceDesc> {
        if !self.is_alive(handle) {
            return None;
        }
        self.slots[handle.index as usize].desc.as_ref()
    }

    /// Rename a resource, for editor-facing labels.
    pub fn set_name(&mut self, handle: ResourceHandle, name: impl Into<String>) -> bool {
        if !self.is_alive(handle) {
            return false;
        }
        self.slots[handle.index as usize].name = name.into();
        true
    }

    /// A resource's name, or a placeholder when it has none.
    pub fn name(&self, handle: ResourceHandle) -> &str {
        if !self.is_alive(handle) {
            return "<dead>";
        }
        &self.slots[handle.index as usize].name
    }

    /// A resource's version, bumped on every change.
    pub fn version(&self, handle: ResourceHandle) -> u64 {
        if handle.index >= self.slots.len() as u32 {
            return 0;
        }
        self.slots[handle.index as usize].version
    }

    /// Number of live resources.
    pub fn len(&self) -> usize {
        self.live
    }

    /// True when no resources are live.
    pub fn is_empty(&self) -> bool {
        self.live == 0
    }

    /// Every live handle.
    pub fn handles(&self) -> impl Iterator<Item = ResourceHandle> + '_ {
        self.slots.iter().enumerate().filter_map(|(i, s)| {
            s.desc.as_ref().map(|_| ResourceHandle {
                index: i as u32,
                generation: s.generation,
            })
        })
    }

    /// Total bytes of all live transient resources, a rough budget figure.
    pub fn transient_bytes(&self) -> u64 {
        self.handles()
            .filter(|h| !self.is_imported(*h))
            .filter_map(|h| self.desc(h))
            .filter(|d| d.is_transient())
            .map(|d| d.size())
            .sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn desc(size: u64) -> ResourceDesc {
        ResourceDesc::Buffer { size }
    }

    #[test]
    fn create_returns_live_handle() {
        let mut r = ResourceRegistry::new();
        let h = r.create(desc(64), "ubo");
        assert!(r.is_alive(h));
        assert_eq!(r.name(h), "ubo");
        assert_eq!(r.len(), 1);
    }

    #[test]
    fn desc_round_trips() {
        let mut r = ResourceRegistry::new();
        let h = r.create(desc(128), "b");
        assert_eq!(r.desc(h), Some(&desc(128)));
    }

    #[test]
    fn free_returns_desc_and_deactivates() {
        let mut r = ResourceRegistry::new();
        let h = r.create(desc(128), "b");
        assert_eq!(r.free(h), Some(desc(128)));
        assert!(!r.is_alive(h));
        assert_eq!(r.free(h), None, "double free must be a no-op");
    }

    #[test]
    fn recycled_slot_gets_new_generation() {
        let mut r = ResourceRegistry::new();
        let a = r.create(desc(1), "a");
        r.free(a);
        let b = r.create(desc(1), "b");
        assert_eq!(a.index(), b.index(), "slot should be reused");
        assert_ne!(a.generation(), b.generation());
    }

    #[test]
    fn stale_handle_does_not_resolve_after_recycle() {
        let mut r = ResourceRegistry::new();
        let a = r.create(desc(1), "a");
        r.free(a);
        let b = r.create(desc(1), "b");
        assert!(!r.is_alive(a), "stale handle must not see the new resource");
        assert!(r.is_alive(b));
        assert_eq!(r.name(a), "<dead>");
    }

    #[test]
    fn stale_free_does_not_kill_the_new_resource() {
        let mut r = ResourceRegistry::new();
        let a = r.create(desc(1), "a");
        r.free(a);
        let b = r.create(desc(1), "b");
        assert_eq!(r.free(a), None);
        assert!(
            r.is_alive(b),
            "a stale free must not destroy the recycled resource"
        );
    }

    #[test]
    fn version_bumps_on_recreate() {
        let mut r = ResourceRegistry::new();
        let h = r.create(desc(1), "a");
        let v0 = r.version(h);
        r.free(h);
        let h2 = r.create(desc(1), "c");
        assert!(
            r.version(h2) > v0,
            "a recycled slot must not look unchanged"
        );
    }

    #[test]
    fn rename_does_not_bump_version() {
        // Version answers "is my cached view of this resource stale", and a
        // label change does not invalidate a view.
        let mut r = ResourceRegistry::new();
        let h = r.create(desc(1), "a");
        let v0 = r.version(h);
        r.set_name(h, "renamed");
        assert_eq!(r.version(h), v0);
    }

    #[test]
    fn rename_updates_name() {
        let mut r = ResourceRegistry::new();
        let h = r.create(desc(1), "a");
        assert!(r.set_name(h, "b"));
        assert_eq!(r.name(h), "b");
    }

    #[test]
    fn rename_on_dead_handle_fails() {
        let mut r = ResourceRegistry::new();
        let h = r.create(desc(1), "a");
        r.free(h);
        assert!(!r.set_name(h, "b"));
    }

    #[test]
    fn import_is_marked_and_not_transient() {
        let mut r = ResourceRegistry::new();
        let h = r.import(desc(1024), "swapchain");
        assert!(r.is_imported(h));
        assert!(r.is_alive(h));
        assert_eq!(r.desc(h), Some(&desc(1024)));
    }

    #[test]
    fn created_resources_are_not_imported() {
        let mut r = ResourceRegistry::new();
        let h = r.create(desc(1), "own");
        assert!(!r.is_imported(h));
    }

    #[test]
    fn transient_bytes_sums_only_transients() {
        let mut r = ResourceRegistry::new();
        r.create(desc(100), "plain-buffer");
        r.create(
            ResourceDesc::Transient {
                size: 256,
                format: 0,
            },
            "color",
        );
        r.create(
            ResourceDesc::Transient {
                size: 512,
                format: 1,
            },
            "depth",
        );
        assert_eq!(r.transient_bytes(), 768);
    }

    #[test]
    fn texture_desc_size_uses_four_bytes_per_texel() {
        let d = ResourceDesc::Texture {
            width: 64,
            height: 32,
            format: 0,
        };
        assert_eq!(d.size(), 64 * 32 * 4);
        assert!(!d.is_transient());
    }

    #[test]
    fn desc_labels() {
        assert_eq!(desc(1).label(), "buffer");
        assert_eq!(
            ResourceDesc::Texture {
                width: 1,
                height: 1,
                format: 0
            }
            .label(),
            "texture"
        );
        assert_eq!(
            ResourceDesc::Transient { size: 1, format: 0 }.label(),
            "transient"
        );
    }

    #[test]
    fn handles_lists_only_live() {
        let mut r = ResourceRegistry::new();
        let a = r.create(desc(1), "a");
        let b = r.create(desc(1), "b");
        r.free(a);
        let live: Vec<_> = r.handles().collect();
        assert_eq!(live, vec![b]);
    }

    #[test]
    fn many_allocations_recycle_slots() {
        let mut r = ResourceRegistry::new();
        let mut all = Vec::new();
        for i in 0..100 {
            all.push(r.create(desc(i), format!("r{i}")));
        }
        assert_eq!(r.len(), 100);
        for h in &all {
            assert!(r.is_alive(*h));
        }
        for h in &all {
            r.free(*h);
        }
        assert!(r.is_empty());
        assert_eq!(r.slots.len(), 100, "slots are recycled, not reallocated");
    }

    #[test]
    fn out_of_range_handle_is_not_alive() {
        let r = ResourceRegistry::new();
        assert!(!r.is_alive(vibe_rhi::ResourceHandle {
            index: 999,
            generation: 0
        }));
        assert_eq!(
            r.desc(vibe_rhi::ResourceHandle {
                index: 999,
                generation: 0
            }),
            None
        );
        assert_eq!(
            r.version(vibe_rhi::ResourceHandle {
                index: 999,
                generation: 0
            }),
            0
        );
    }

    #[test]
    fn empty_registry_reports_empty() {
        let r = ResourceRegistry::new();
        assert!(r.is_empty());
        assert_eq!(r.len(), 0);
        assert_eq!(r.transient_bytes(), 0);
    }
}
