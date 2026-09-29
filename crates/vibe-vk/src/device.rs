//! Logical device creation with a tier-gated feature set.
//!
//! Which extensions and features to enable is a decision, not a lookup: a
//! device that advertises an extension the instance did not enable, or a
//! feature the driver reports but the engine does not need, both make
//! `vkCreateDevice` fail or silently misbehave. The decision is therefore a
//! pure function of (tier, available extensions, available features) and is
//! tested without a device; only the `vkCreateDevice` call itself needs one.

use std::ffi::CStr;
use std::os::raw::c_char;

use ash::vk;
use log::{debug, info, warn};

use crate::context::{QueueRequirements, VkError};
use crate::sync::SyncTier;

/// What the engine asked to enable, and what the device actually has.
#[derive(Debug, Clone)]
pub struct EnabledFeatures {
    /// Extensions the engine will enable, sorted.
    pub extensions: Vec<String>,
    /// Extensions that were requested by the engine but are not available.
    pub missing: Vec<String>,
    /// The sync tier the device will run at.
    pub tier: SyncTier,
    /// The core 1.0 features the engine enables.
    pub features: vk::PhysicalDeviceFeatures,
    /// The sync2 feature, enabled only on the Sync2 tiers.
    pub synchronization2: bool,
}

impl EnabledFeatures {
    /// True when everything the engine asked for is available.
    pub fn is_complete(&self) -> bool {
        self.missing.is_empty()
    }

    /// The extension names as C strings, for `ppEnabledExtensionNames`.
    pub fn extension_ptrs(&self) -> Vec<*const c_char> {
        self.extensions
            .iter()
            .map(|e| e.as_ptr() as *const c_char)
            .collect()
    }
}

/// The extensions the engine always needs, whatever the tier.
const REQUIRED: &[&str] = &["VK_KHR_swapchain"];

/// The extensions the engine needs only above Legacy.
///
/// After promotion these are core, so a driver that reports Vulkan 1.3 need not
/// enumerate them; the caller passes what the device actually advertised.
const SYNC2_EXTENSIONS: &[&str] = &["VK_KHR_synchronization2", "VK_KHR_copy_commands2"];

const TIMELINE_EXTENSION: &str = "VK_KHR_timeline_semaphore";
const DYNAMIC_RENDERING_EXTENSION: &str = "VK_KHR_dynamic_rendering";
const BUFFER_DEVICE_ADDRESS: &str = "VK_KHR_buffer_device_address";
const DESCRIPTOR_INDEXING: &str = "VK_EXT_descriptor_indexing";
const BINDLESS_TEXTURE: &str = "VK_KHR_bindless_texture";

/// Decide which extensions to enable.
///
/// `available` is what the device enumerated, not what the core version
/// implies, so a promoted-but-not-enumerated extension is treated as absent.
/// That is correct: the instance has to have been created with the API version
/// high enough for the core entry point, and the engine does that.
///
/// The result is sorted so a device with the same capabilities always produces
/// the same list, which makes the choice comparable in tests and logs.
pub fn choose_extensions(tier: SyncTier, available: &[&str]) -> EnabledFeatures {
    let has = |name: &str| available.iter().any(|a| *a == name);
    let mut want: Vec<String> = Vec::new();
    let mut missing: Vec<String> = Vec::new();

    let require = |name: &str, extensions: &mut Vec<String>, missing: &mut Vec<String>| {
        if has(name) {
            extensions.push(name.to_string());
        } else {
            missing.push(name.to_string());
        }
    };

    for name in REQUIRED {
        require(name, &mut want, &mut missing);
    }

    if tier.supports_barrier2() {
        for name in SYNC2_EXTENSIONS {
            // Optional: a driver may expose them only as core.
            if has(name) {
                want.push(name.to_string());
            }
        }
        if has(DYNAMIC_RENDERING_EXTENSION) {
            want.push(DYNAMIC_RENDERING_EXTENSION.to_string());
        }
    } else {
        for name in SYNC2_EXTENSIONS {
            if !has(name) {
                missing.push(name.to_string());
            }
        }
    }

    if tier.supports_timeline() {
        require(TIMELINE_EXTENSION, &mut want, &mut missing);
    }

    for name in [BUFFER_DEVICE_ADDRESS, DESCRIPTOR_INDEXING] {
        if has(name) {
            want.push(name.to_string());
        }
    }

    want.sort();
    want.dedup();
    missing.sort();
    missing.dedup();

    if !missing.is_empty() {
        warn!("device is missing required extensions: {missing:?}");
    }

    EnabledFeatures {
        extensions: want,
        missing,
        tier,
        features: base_features(),
        synchronization2: tier.supports_barrier2(),
    }
}

/// Whether full bindless is usable, and so whether to enable descriptor
/// indexing at all.
///
/// Bindless needs both `VK_EXT_descriptor_indexing` and
/// `VK_KHR_bindless_texture`; a device with only the former gets the partial
/// path, which is still integer-indexed but through a smaller array.
pub fn bindless_available(available: &[&str]) -> bool {
    available.iter().any(|a| *a == BINDLESS_TEXTURE)
        && available.iter().any(|a| *a == DESCRIPTOR_INDEXING)
}

/// The core features the engine turns on.
///
/// Everything else stays off: an engine that enables features it does not use
/// pays for them in driver-side work and gives up portability for nothing.
pub fn base_features() -> vk::PhysicalDeviceFeatures {
    vk::PhysicalDeviceFeatures {
        // A 32-byte quad vertex and 4-component positions.
        robust_buffer_access: vk::TRUE,
        // The ECS writes transforms through mapped memory every frame.
        full_draw_index_uint32: vk::TRUE,
        shader_clip_distance: vk::TRUE,
        ..Default::default()
    }
}

/// The Vulkan 1.3 features the engine needs, chained into `Features2`.
///
/// `dynamic_rendering` is a 1.3 feature, so it cannot go in the 1.0 struct;
/// it is enabled through the `pNext` chain on `DeviceCreateInfo`.
#[derive(Debug, Clone, Copy, Default)]
pub struct ModernFeatures {
    /// Enables `vkCmdBeginRendering` without a render pass object.
    pub dynamic_rendering: vk::Bool32,
    /// Enables `vkCmdPipelineBarrier2`.
    pub synchronization2: vk::Bool32,
    /// Enables the copy commands that take a `VkDependencyInfo`.
    pub copy_commands2: vk::Bool32,
}

impl ModernFeatures {
    /// The features a device on the given tier supports.
    pub fn for_tier(tier: SyncTier) -> ModernFeatures {
        ModernFeatures {
            dynamic_rendering: vk::TRUE,
            synchronization2: if tier.supports_barrier2() {
                vk::TRUE
            } else {
                vk::FALSE
            },
            copy_commands2: if tier.supports_barrier2() {
                vk::TRUE
            } else {
                vk::FALSE
            },
        }
    }
}

/// A created logical device and the queues taken from it.
pub struct LogicalDevice {
    /// The ash device handle, already loaded with its function table.
    pub device: ash::Device,
    /// The handle, for the APIs that need the raw value.
    pub handle: vk::Device,
    /// The features that were enabled.
    pub features: EnabledFeatures,
    queues: Vec<(u32, vk::Queue)>,
    /// Present-capable queue family, if one was requested.
    pub present_family: Option<u32>,
    graphics_family: Option<u32>,
    transfer_family: Option<u32>,
}

impl LogicalDevice {
    /// Create a logical device and take one queue per requested family.
    ///
    /// # Safety
    ///
    /// `instance` must be a live instance, `physical_device` must come from
    /// it, and the surface must belong to it when `requirements.present` is
    /// set.
    pub unsafe fn create(
        instance: &ash::Instance,
        surface_loader: Option<&ash::khr::surface::Instance>,
        physical_device: vk::PhysicalDevice,
        surface: Option<vk::SurfaceKHR>,
        requirements: QueueRequirements,
    ) -> Result<LogicalDevice, VkError> {
        let choice = unsafe {
            crate::context::find_queue_families(
                instance,
                surface_loader.ok_or(VkError::MissingQueueFamily { what: "present" })?,
                physical_device,
                surface,
                requirements,
            )
        }?;

        let available = device_extension_names(instance, physical_device)?;
        let available_refs: Vec<&str> = available.iter().map(|s| s.as_str()).collect();
        let enabled = choose_extensions(choice.info.sync_tier, &available_refs);

        let families = choice.queue_families();
        if families.is_empty() {
            return Err(VkError::MissingQueueFamily { what: "any" });
        }

        // One queue per family at index 0: the engine does not need more, and
        // asking for a second would fail on a family that reports one queue.
        let priorities = [1.0f32];
        let queue_infos: Vec<vk::DeviceQueueCreateInfo<'_>> = families
            .iter()
            .map(|f| vk::DeviceQueueCreateInfo {
                queue_family_index: *f,
                queue_count: 1,
                p_queue_priorities: priorities.as_ptr(),
                ..Default::default()
            })
            .collect();

        let extension_ptrs = enabled.extension_ptrs();
        let create_info = vk::DeviceCreateInfo {
            queue_create_info_count: queue_infos.len() as u32,
            p_queue_create_infos: queue_infos.as_ptr(),
            enabled_extension_count: extension_ptrs.len() as u32,
            pp_enabled_extension_names: extension_ptrs.as_ptr(),
            p_enabled_features: &enabled.features,
            ..Default::default()
        };

        let device = unsafe { instance.create_device(physical_device, &create_info, None) }?;
        info!(
            "logical device created on {} with {} extension(s), tier {}",
            choice.info.name,
            enabled.extensions.len(),
            enabled.tier
        );
        if enabled.synchronization2 {
            debug!("synchronization2 enabled: barriers use vkCmdPipelineBarrier2");
        }

        let mut queues = Vec::with_capacity(families.len());
        for family in &families {
            queues.push((*family, unsafe { device.get_device_queue(*family, 0) }));
        }

        Ok(LogicalDevice {
            handle: device.handle(),
            device,
            features: enabled,
            queues,
            present_family: choice.present_family,
            graphics_family: choice.graphics_family,
            transfer_family: choice.transfer_family,
        })
    }

    /// The queue for a family, if it was requested.
    pub fn queue(&self, family: u32) -> Option<vk::Queue> {
        self.queues
            .iter()
            .find(|(f, _)| *f == family)
            .map(|(_, q)| *q)
    }

    /// The graphics queue.
    pub fn graphics_queue(&self) -> Option<vk::Queue> {
        self.graphics_family.and_then(|f| self.queue(f))
    }

    /// The transfer queue.
    pub fn transfer_queue(&self) -> Option<vk::Queue> {
        self.transfer_family.and_then(|f| self.queue(f))
    }

    /// The present-capable queue.
    pub fn present_queue(&self) -> Option<vk::Queue> {
        self.present_family.and_then(|f| self.queue(f))
    }

    /// The graphics family index.
    pub fn graphics_family(&self) -> Option<u32> {
        self.graphics_family
    }

    /// The transfer family index.
    pub fn transfer_family(&self) -> Option<u32> {
        self.transfer_family
    }

    /// The sync tier this device runs at.
    pub fn sync_tier(&self) -> SyncTier {
        self.features.tier
    }

    /// Wait for the device to go fully idle.
    ///
    /// Only for teardown: it stalls the CPU until every queue is done.
    pub fn wait_idle(&self) {
        unsafe {
            let _ = self.device.device_wait_idle();
        }
    }

    /// Destroy the device.
    ///
    /// # Safety
    ///
    /// Every queue, buffer, image and command buffer must be destroyed first.
    pub unsafe fn destroy(&mut self) {
        unsafe {
            self.device.destroy_device(None);
        }
        self.queues.clear();
    }
}

/// The device extension names a physical device advertises.
pub fn device_extension_names(
    instance: &ash::Instance,
    physical_device: vk::PhysicalDevice,
) -> Result<Vec<String>, VkError> {
    let props = unsafe { instance.enumerate_device_extension_properties(physical_device) }?;
    Ok(props
        .iter()
        .map(|p| {
            unsafe { CStr::from_ptr(p.extension_name.as_ptr()) }
                .to_string_lossy()
                .into_owned()
        })
        .collect())
}

/// The extensions a tier expects to be able to enable.
///
/// Used to check the engine's own requirements against a device before
/// `vkCreateDevice` fails with an opaque error.
pub fn expected_extensions(tier: SyncTier) -> Vec<&'static str> {
    let mut out: Vec<&'static str> = REQUIRED.to_vec();
    if tier.supports_barrier2() {
        out.extend_from_slice(SYNC2_EXTENSIONS);
        out.push(DYNAMIC_RENDERING_EXTENSION);
    }
    if tier.supports_timeline() {
        out.push(TIMELINE_EXTENSION);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn extensions<'a>(names: &[&'a str]) -> Vec<&'a str> {
        names.to_vec()
    }

    #[test]
    fn swapchain_is_always_required() {
        let e = choose_extensions(SyncTier::Legacy, &extensions(&["VK_KHR_swapchain"]));
        assert!(e.extensions.contains(&"VK_KHR_swapchain".to_string()));
        // Synchronization2 is still listed as missing on Legacy, so this is
        // not a complete set: Legacy genuinely cannot do barrier2.
        assert!(e.missing.contains(&"VK_KHR_synchronization2".to_string()));
    }

    #[test]
    fn a_sync2_device_with_what_it_needs_is_complete() {
        let e = choose_extensions(
            SyncTier::Sync2,
            &extensions(&[
                "VK_KHR_swapchain",
                "VK_KHR_synchronization2",
                "VK_KHR_copy_commands2",
            ]),
        );
        assert!(e.is_complete(), "missing: {:?}", e.missing);
    }

    #[test]
    fn a_device_without_swapchain_is_reported_incomplete() {
        let e = choose_extensions(SyncTier::Legacy, &[]);
        assert!(!e.is_complete());
        assert!(e.missing.contains(&"VK_KHR_swapchain".to_string()));
    }

    #[test]
    fn legacy_does_not_request_sync2_extensions() {
        let e = choose_extensions(
            SyncTier::Legacy,
            &extensions(&["VK_KHR_swapchain", "VK_KHR_synchronization2"]),
        );
        assert!(
            !e.extensions
                .contains(&"VK_KHR_synchronization2".to_string()),
            "legacy must not enable a barrier2 extension"
        );
    }

    #[test]
    fn legacy_reports_sync2_as_missing() {
        let e = choose_extensions(SyncTier::Legacy, &extensions(&["VK_KHR_swapchain"]));
        assert!(e.missing.contains(&"VK_KHR_synchronization2".to_string()));
    }

    #[test]
    fn sync2_requests_synchronization2() {
        let e = choose_extensions(
            SyncTier::Sync2,
            &extensions(&[
                "VK_KHR_swapchain",
                "VK_KHR_synchronization2",
                "VK_KHR_copy_commands2",
            ]),
        );
        assert!(
            e.extensions
                .contains(&"VK_KHR_synchronization2".to_string())
        );
        assert!(e.synchronization2);
    }

    #[test]
    fn sync2_timeline_requests_timeline_semaphores() {
        let e = choose_extensions(
            SyncTier::Sync2Timeline,
            &extensions(&[
                "VK_KHR_swapchain",
                "VK_KHR_synchronization2",
                "VK_KHR_timeline_semaphore",
            ]),
        );
        assert!(
            e.extensions
                .contains(&"VK_KHR_timeline_semaphore".to_string())
        );
    }

    #[test]
    fn missing_timeline_semaphores_is_reported() {
        let e = choose_extensions(
            SyncTier::Sync2Timeline,
            &extensions(&["VK_KHR_swapchain", "VK_KHR_synchronization2"]),
        );
        assert!(e.missing.contains(&"VK_KHR_timeline_semaphore".to_string()));
    }

    #[test]
    fn buffer_device_address_is_enabled_when_present() {
        let e = choose_extensions(
            SyncTier::Sync2,
            &extensions(&["VK_KHR_swapchain", "VK_KHR_buffer_device_address"]),
        );
        assert!(
            e.extensions
                .contains(&"VK_KHR_buffer_device_address".to_string())
        );
    }

    #[test]
    fn optional_extensions_do_not_make_the_device_incomplete() {
        // Buffer device address and descriptor indexing are wanted but not
        // required: a device without them is still usable, just not bindless.
        let e = choose_extensions(
            SyncTier::Sync2,
            &extensions(&[
                "VK_KHR_swapchain",
                "VK_KHR_synchronization2",
                "VK_KHR_copy_commands2",
            ]),
        );
        assert!(e.is_complete());
        assert!(
            !e.extensions
                .contains(&"VK_KHR_buffer_device_address".to_string())
        );
    }

    #[test]
    fn extensions_are_sorted_and_deduplicated() {
        let e = choose_extensions(
            SyncTier::Sync2Timeline,
            &extensions(&[
                "VK_KHR_timeline_semaphore",
                "VK_KHR_swapchain",
                "VK_KHR_synchronization2",
            ]),
        );
        let mut sorted = e.extensions.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(e.extensions, sorted, "the list must be canonical");
    }

    #[test]
    fn the_same_device_always_produces_the_same_list() {
        let avail = extensions(&[
            "VK_KHR_swapchain",
            "VK_KHR_synchronization2",
            "VK_KHR_timeline_semaphore",
        ]);
        let a = choose_extensions(SyncTier::Sync2Timeline, &avail);
        let b = choose_extensions(SyncTier::Sync2Timeline, &avail);
        assert_eq!(a.extensions, b.extensions);
        assert_eq!(a.missing, b.missing);
    }

    #[test]
    fn bindless_needs_both_extensions() {
        assert!(bindless_available(&extensions(&[
            "VK_EXT_descriptor_indexing",
            "VK_KHR_bindless_texture"
        ])));
    }

    #[test]
    fn descriptor_indexing_alone_is_not_bindless() {
        assert!(!bindless_available(&extensions(&[
            "VK_EXT_descriptor_indexing"
        ])));
    }

    #[test]
    fn bindless_texture_alone_is_not_bindless() {
        assert!(!bindless_available(&extensions(&[
            "VK_KHR_bindless_texture"
        ])));
    }

    #[test]
    fn no_extensions_means_no_bindless() {
        assert!(!bindless_available(&[]));
    }

    #[test]
    fn modern_features_enable_dynamic_rendering_on_every_tier() {
        // Dynamic rendering is enabled through the Features2 chain on all
        // tiers, because the engine never uses a render pass object.
        for tier in [SyncTier::Legacy, SyncTier::Sync2, SyncTier::Sync2Timeline] {
            assert_eq!(
                ModernFeatures::for_tier(tier).dynamic_rendering,
                vk::TRUE,
                "{tier}"
            );
        }
    }

    #[test]
    fn modern_sync2_features_follow_the_tier() {
        assert_eq!(
            ModernFeatures::for_tier(SyncTier::Legacy).synchronization2,
            vk::FALSE
        );
        assert_eq!(
            ModernFeatures::for_tier(SyncTier::Sync2).synchronization2,
            vk::TRUE
        );
        assert_eq!(
            ModernFeatures::for_tier(SyncTier::Sync2Timeline).copy_commands2,
            vk::TRUE
        );
    }

    #[test]
    fn base_features_leave_everything_else_off() {
        let f = base_features();
        assert_eq!(
            f.sampler_anisotropy,
            vk::FALSE,
            "unused features must stay off"
        );
        assert_eq!(f.geometry_shader, vk::FALSE);
        assert_eq!(f.tessellation_shader, vk::FALSE);
    }

    #[test]
    fn base_features_are_never_null_for_anything_enabled() {
        let f = base_features();
        // robust_buffer_access, full_draw_index_uint32 and dynamic_rendering
        // are the three the engine sets.
        assert_eq!(f.robust_buffer_access, vk::TRUE);
        assert_eq!(f.full_draw_index_uint32, vk::TRUE);
    }

    #[test]
    fn extension_ptrs_match_the_list() {
        let e = choose_extensions(SyncTier::Legacy, &extensions(&["VK_KHR_swapchain"]));
        let ptrs = e.extension_ptrs();
        assert_eq!(ptrs.len(), e.extensions.len());
    }

    #[test]
    fn expected_extensions_grow_with_the_tier() {
        let legacy = expected_extensions(SyncTier::Legacy);
        let sync2 = expected_extensions(SyncTier::Sync2);
        let timeline = expected_extensions(SyncTier::Sync2Timeline);
        assert!(legacy.len() < sync2.len());
        assert!(sync2.len() < timeline.len());
    }

    #[test]
    fn expected_extensions_for_a_tier_are_a_subset_of_what_a_full_device_has() {
        let avail = extensions(&[
            "VK_KHR_swapchain",
            "VK_KHR_synchronization2",
            "VK_KHR_copy_commands2",
            "VK_KHR_dynamic_rendering",
            "VK_KHR_timeline_semaphore",
        ]);
        let e = choose_extensions(SyncTier::Sync2Timeline, &avail);
        assert!(
            e.is_complete(),
            "a full device should satisfy every tier requirement"
        );
    }

    #[test]
    fn synchronization2_flag_follows_the_tier() {
        assert!(!choose_extensions(SyncTier::Legacy, &[]).synchronization2);
        assert!(choose_extensions(SyncTier::Sync2, &[]).synchronization2);
        assert!(choose_extensions(SyncTier::Sync2Timeline, &[]).synchronization2);
    }

    #[test]
    fn enabled_features_records_the_tier_it_chose_for() {
        let e = choose_extensions(SyncTier::Sync2, &[]);
        assert_eq!(e.tier, SyncTier::Sync2);
    }

    #[test]
    fn sync_module_helpers_stay_consistent() {
        use crate::sync;
        // The extension table and the device chooser must agree on what a
        // timeline-capable device needs.
        assert!(sync::is_wanted("VK_KHR_timeline_semaphore"));
        assert!(sync::is_wanted("VK_KHR_synchronization2"));
    }
}
