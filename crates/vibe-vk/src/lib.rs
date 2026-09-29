//! Vulkan backend for vibeEngine.
//!
//! Layered: [`sync`] is pure logic over API versions and extension lists and
//! needs no device, [`context`] does the loader and device work, and the
//! renderer builds on top.

pub mod barrier;
pub mod context;
pub mod frame;
pub mod graph_bridge;
pub mod swapchain;
pub mod sync;

pub use barrier::{
    Access, Barrier, BarrierEncoder, ResourceKind, Stage, image_to_shader_read, upload_barriers,
};
pub use context::{
    Entry, PhysicalDeviceChoice, PhysicalDeviceInfo, QueueRequirements, VkError, create_instance,
    describe_device, enumerate_devices, find_queue_families, load_entry, score_device,
    select_device,
};
pub use frame::{DeferredDestructionQueue, FrameConfig, FramePacer, default_frames_in_flight};
pub use graph_bridge::{
    BackendObject, ObjectResolver, ResourceRole, encode_graph, encode_graph_for_tier, translate,
};
pub use swapchain::{
    AcquireResult, ResizeMode, Swapchain, choose_image_count, choose_present_mode,
    choose_surface_format, clamp_extent, classify_extent,
};
pub use sync::{
    ApiVersion, DECLARED_EXTENSIONS, ExtensionInfo, SyncTier, api_version, detect_bindless_support,
    detect_sync_tier, tier_for_loader, version_string,
};

/// The backend name used in `vibe-rhi` dispatch.
pub const BACKEND: vibe_rhi::Backend = vibe_rhi::Backend::Vulkan;
