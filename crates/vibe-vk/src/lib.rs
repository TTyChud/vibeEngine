//! Vulkan backend for vibeEngine.
//!
//! Layered: [`sync`] is pure logic over API versions and extension lists and
//! needs no device, [`context`] does the loader and device work, and the
//! renderer builds on top.

pub mod context;
pub mod sync;

pub use context::{
    Entry, PhysicalDeviceChoice, PhysicalDeviceInfo, QueueRequirements, VkError, create_instance,
    describe_device, enumerate_devices, find_queue_families, load_entry, score_device,
    select_device,
};
pub use sync::{
    ApiVersion, DECLARED_EXTENSIONS, ExtensionInfo, SyncTier, api_version, detect_bindless_support,
    detect_sync_tier, tier_for_loader, version_string,
};

/// The backend name used in `vibe-rhi` dispatch.
pub const BACKEND: vibe_rhi::Backend = vibe_rhi::Backend::Vulkan;
