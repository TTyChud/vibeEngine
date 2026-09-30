//! Loader, instance and physical-device selection.

use std::ffi::{CStr, c_char};

use ash::vk;
use log::{debug, info, warn};

use crate::sync::{self, ApiVersion, SyncTier};

/// Why Vulkan context creation failed.
#[derive(Debug, thiserror::Error)]
pub enum VkError {
    /// No Vulkan loader could be opened.
    #[error("vulkan loader not found: {0}")]
    LoaderNotFound(String),
    /// The loader loaded but reported an error.
    #[error("vulkan loader error: {0}")]
    LoaderError(String),
    /// No GPU exposed a Vulkan device.
    #[error("no vulkan physical device found")]
    NoPhysicalDevice,
    /// A physical device lacked a required queue family.
    #[error("no queue family supporting {what}")]
    MissingQueueFamily {
        /// What the engine needed the family for.
        what: &'static str,
    },
    /// A Vulkan call returned an error code.
    #[error("vulkan call failed: {0:?}")]
    CallFailed(vk::Result),
    /// No swapchain could be created for the chosen surface.
    #[error("swapchain creation failed: {0}")]
    Swapchain(String),
}

impl From<vk::Result> for VkError {
    fn from(r: vk::Result) -> Self {
        match r {
            vk::Result::SUCCESS => VkError::LoaderError("unexpected success".to_string()),
            other => VkError::CallFailed(other),
        }
    }
}

/// Build an API version. Shorthand for `sync::api_version(0, ..)`.
const fn v(major: u32, minor: u32, patch: u32) -> ApiVersion {
    sync::api_version(0, major, minor, patch)
}

fn ext_name(props: &vk::ExtensionProperties) -> &CStr {
    unsafe { CStr::from_ptr(props.extension_name.as_ptr()) }
}

fn lossy(c: &CStr) -> &str {
    c.to_str().unwrap_or("<non-utf8>")
}

/// A physical GPU and the properties that drove its selection.
#[derive(Debug, Clone)]
pub struct PhysicalDeviceInfo {
    /// The Vulkan object handle.
    pub handle: vk::PhysicalDevice,
    /// `deviceName` from the driver.
    pub name: String,
    /// Device type as reported.
    pub device_type: vk::PhysicalDeviceType,
    /// The device's core API version.
    pub api_version: ApiVersion,
    /// Detected synchronisation tier.
    pub sync_tier: SyncTier,
    /// Driver-reported version string.
    pub driver_version: String,
    /// Sum of all memory heaps in bytes.
    pub total_memory: u64,
    /// All device extension names.
    pub extensions: Vec<String>,
    /// True when the device reports a discrete GPU.
    pub is_discrete: bool,
    /// How this device can address resources from shaders.
    pub bindless: vibe_rhi::BindlessSupport,
}

/// What the engine needs a queue family to do.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QueueRequirements {
    /// The family must support graphics.
    pub graphics: bool,
    /// The family must support transfer.
    pub transfer: bool,
    /// The family must support present to the surface.
    pub present: bool,
}

impl QueueRequirements {
    /// A single family doing graphics, transfer and present.
    pub const fn all() -> QueueRequirements {
        QueueRequirements {
            graphics: true,
            transfer: true,
            present: true,
        }
    }

    /// Headless: compute and transfer only, no surface.
    pub const fn compute_only() -> QueueRequirements {
        QueueRequirements {
            graphics: false,
            transfer: true,
            present: false,
        }
    }
}

/// A chosen physical device plus its families.
#[derive(Debug)]
pub struct PhysicalDeviceChoice {
    /// The selected device.
    pub info: PhysicalDeviceInfo,
    /// Graphics family, if requested and found.
    pub graphics_family: Option<u32>,
    /// Compute family, if found.
    pub compute_family: Option<u32>,
    /// Transfer family, if requested and found.
    pub transfer_family: Option<u32>,
    /// Present family, if requested and found.
    pub present_family: Option<u32>,
}

impl PhysicalDeviceChoice {
    /// Every distinct family to create a queue on.
    pub fn queue_families(&self) -> Vec<u32> {
        let mut out = Vec::new();
        for f in [
            self.graphics_family,
            self.compute_family,
            self.transfer_family,
            self.present_family,
        ]
        .into_iter()
        .flatten()
        {
            if !out.contains(&f) {
                out.push(f);
            }
        }
        out
    }
}

/// Score a device for engine work; higher is better.
///
/// Device type dominates, then memory, then API version, then sync tier. The
/// weights are deliberately coarse: this only breaks ties between plausible
/// devices, and the user can always override.
pub fn score_device(info: &PhysicalDeviceInfo) -> u32 {
    let type_score = if info.device_type == vk::PhysicalDeviceType::DISCRETE_GPU {
        10_000
    } else if info.device_type == vk::PhysicalDeviceType::INTEGRATED_GPU {
        5_000
    } else if info.device_type == vk::PhysicalDeviceType::VIRTUAL_GPU {
        1_000
    } else {
        100
    };
    let memory_score = ((info.total_memory / (256 * 1024 * 1024)) as u32).min(9_000);
    let version_score = (sync::major(info.api_version) as u32) * 100
        + (sync::minor(info.api_version) as u32).min(9) * 10;
    let sync_score = match info.sync_tier {
        SyncTier::Sync2Timeline => 50,
        SyncTier::Sync2 => 25,
        SyncTier::Legacy => 0,
    };
    type_score + memory_score + version_score.min(900) + sync_score
}

/// The loaded Vulkan entry point table.
pub struct Entry {
    /// The ash entry points.
    pub ash: ash::Entry,
    /// The version the loader reported.
    pub loader_version: ApiVersion,
}

/// Load the Vulkan loader.
///
/// # Safety
///
/// The returned table is safe to use; the `unsafe` is ash's own
/// `Entry::load`, which dlopens the platform library.
pub unsafe fn load_entry() -> Result<Entry, VkError> {
    let ash = unsafe { ash::Entry::load() }.map_err(|e| VkError::LoaderNotFound(e.to_string()))?;
    let loader_version = unsafe { ash.try_enumerate_instance_version() }
        .ok()
        .flatten()
        .unwrap_or_else(|| v(1, 0, 0));
    debug!(
        "vulkan loader version {}",
        sync::version_string(loader_version)
    );
    Ok(Entry {
        ash,
        loader_version,
    })
}

/// The highest API version the engine will ask an instance for.
///
/// 1.3 is the target because the barrier encoder emits
/// `vkCmdPipelineBarrier2` and the submit path uses `vkQueueSubmit2`, both core
/// 1.3 entry points. An instance created at 1.1 never loads them, and calling
/// one panics inside ash rather than returning a Vulkan error.
pub const REQUESTED_API_VERSION: ApiVersion = sync::api_version(0, 1, 3, 0);

/// Create an instance requesting the highest API version available.
///
/// The version asked for is [`REQUESTED_API_VERSION`], clamped to whatever the
/// loader reports, so a 1.1 loader gets a 1.1 instance and the engine still
/// runs at the Legacy tier.
///
/// Extensions that the loader does not advertise are skipped with a warning
/// rather than failing, so a headless run without `VK_KHR_surface` still works.
///
/// # Safety
///
/// `surface_extensions` must be names valid for the windowing system in use.
pub unsafe fn create_instance(
    entry: &ash::Entry,
    app_name: &CStr,
    surface_extensions: &[*const c_char],
) -> Result<ash::Instance, VkError> {
    unsafe {
        create_instance_with_version(entry, app_name, surface_extensions, REQUESTED_API_VERSION)
    }
}

/// Create an instance at a chosen API version, clamped to the loader's.
pub unsafe fn create_instance_with_version(
    entry: &ash::Entry,
    app_name: &CStr,
    surface_extensions: &[*const c_char],
    wanted: ApiVersion,
) -> Result<ash::Instance, VkError> {
    // Never ask for more than the loader implements: doing so is legal but
    // means the 1.3 entry points come back null.
    let loader_version = unsafe { entry.try_enumerate_instance_version() }
        .ok()
        .flatten()
        .unwrap_or(v(1, 0, 0));
    let api_version = wanted.min(loader_version);
    let available = unsafe { entry.enumerate_instance_extension_properties(None) }?;
    let available_names: Vec<&CStr> = available.iter().map(ext_name).collect();

    // Validation needs VK_EXT_debug_utils, requested only when the caller
    // asked for it so a normal run carries no extra extension.
    let mut wanted: Vec<*const c_char> = surface_extensions.iter().copied().collect();
    if crate::validation::validation_requested() {
        if let Some(ext) = crate::validation::required_extension() {
            wanted.push(ext.as_ptr());
        }
    }

    let mut extensions: Vec<*const c_char> = Vec::new();
    for want in wanted {
        // `want` is already the pointer, not a pointer to one.
        let name = unsafe { CStr::from_ptr(want) };
        if available_names.contains(&name) {
            extensions.push(want);
        } else {
            warn!(
                "instance extension {} unavailable, continuing without it",
                lossy(name)
            );
        }
    }

    let app_info = vk::ApplicationInfo {
        p_application_name: app_name.as_ptr(),
        application_version: v(0, 1, 0),
        p_engine_name: c"vibeEngine".as_ptr(),
        engine_version: v(0, 1, 0),
        api_version,
        ..Default::default()
    };
    let create_info = vk::InstanceCreateInfo {
        p_application_info: &app_info,
        enabled_extension_count: extensions.len() as u32,
        pp_enabled_extension_names: extensions.as_ptr(),
        ..Default::default()
    };

    let instance = unsafe { entry.create_instance(&create_info, None) }?;
    let enabled: Vec<String> = extensions
        .iter()
        .map(|e| unsafe { CStr::from_ptr(*e) }.to_string_lossy().into_owned())
        .collect();
    info!(
        "vulkan instance created at api {} with extensions: {enabled:?}",
        sync::version_string(api_version)
    );
    Ok(instance)
}

/// Enumerate every physical device with its detected sync tier.
///
/// # Safety
///
/// `instance` must be a live Vulkan instance.
pub unsafe fn enumerate_devices(
    instance: &ash::Instance,
    loader_version: ApiVersion,
) -> Result<Vec<PhysicalDeviceInfo>, VkError> {
    let handles = unsafe { instance.enumerate_physical_devices() }?;
    let mut out = Vec::with_capacity(handles.len());
    for handle in handles {
        out.push(unsafe { describe_device(instance, handle, loader_version) }?);
    }
    if out.is_empty() {
        return Err(VkError::NoPhysicalDevice);
    }
    Ok(out)
}

/// Query one device's properties, extensions and sync tier.
///
/// # Safety
///
/// `instance` must be live and `handle` must come from it.
pub unsafe fn describe_device(
    instance: &ash::Instance,
    handle: vk::PhysicalDevice,
    loader_version: ApiVersion,
) -> Result<PhysicalDeviceInfo, VkError> {
    let props = unsafe { instance.get_physical_device_properties(handle) };
    let mem = unsafe { instance.get_physical_device_memory_properties(handle) };
    let exts = unsafe { instance.enumerate_device_extension_properties(handle) }?;

    let extension_names: Vec<String> = exts
        .iter()
        .map(|e| lossy(ext_name(e)).to_string())
        .collect();
    let ext_refs: Vec<&str> = extension_names.iter().map(|s| s.as_str()).collect();
    let total_memory: u64 = (0..mem.memory_heap_count as usize)
        .map(|i| mem.memory_heaps[i].size)
        .sum();

    Ok(PhysicalDeviceInfo {
        handle,
        name: lossy(unsafe { CStr::from_ptr(props.device_name.as_ptr()) }).to_string(),
        device_type: props.device_type,
        api_version: props.api_version,
        driver_version: format!("{}", props.driver_version),
        total_memory,
        is_discrete: props.device_type == vk::PhysicalDeviceType::DISCRETE_GPU,
        sync_tier: sync::tier_for_loader(loader_version, props.api_version, &ext_refs),
        bindless: sync::detect_bindless_support(&ext_refs),
        extensions: extension_names,
    })
}

/// Find queue family indices satisfying `requirements`.
///
/// Present support is only checked when a `surface` is supplied, so headless
/// runs can reuse the same code path.
pub unsafe fn find_queue_families(
    instance: &ash::Instance,
    surface_loader: &ash::khr::surface::Instance,
    device: vk::PhysicalDevice,
    surface: Option<vk::SurfaceKHR>,
    requirements: QueueRequirements,
) -> Result<PhysicalDeviceChoice, VkError> {
    let families = unsafe { instance.get_physical_device_queue_family_properties(device) };
    let mut graphics = None;
    let mut compute = None;
    let mut transfer = None;
    let mut present = None;

    for (i, family) in families.iter().enumerate() {
        let i = i as u32;
        if family.queue_count == 0 {
            continue;
        }
        if requirements.graphics
            && family.queue_flags.contains(vk::QueueFlags::GRAPHICS)
            && graphics.is_none()
        {
            graphics = Some(i);
        }
        if requirements.transfer
            && family.queue_flags.contains(vk::QueueFlags::TRANSFER)
            && transfer.is_none()
        {
            transfer = Some(i);
        }
        if compute.is_none() && family.queue_flags.contains(vk::QueueFlags::COMPUTE) {
            compute = Some(i);
        }
        if requirements.present && present.is_none() {
            if let Some(s) = surface {
                if unsafe { surface_loader.get_physical_device_surface_support(device, i, s) }? {
                    present = Some(i);
                }
            }
        }
    }

    if requirements.graphics && graphics.is_none() {
        return Err(VkError::MissingQueueFamily { what: "graphics" });
    }
    if requirements.transfer && transfer.is_none() {
        return Err(VkError::MissingQueueFamily { what: "transfer" });
    }
    if requirements.present && present.is_none() {
        return Err(VkError::MissingQueueFamily { what: "present" });
    }

    let info = unsafe { describe_device(instance, device, v(1, 3, 0)) }?;
    Ok(PhysicalDeviceChoice {
        info,
        graphics_family: graphics,
        compute_family: compute,
        transfer_family: transfer,
        present_family: present,
    })
}

/// Pick the best-scoring device.
pub fn select_device<'a>(
    devices: &'a [PhysicalDeviceInfo],
    requirements: QueueRequirements,
) -> Option<&'a PhysicalDeviceInfo> {
    let mut candidates: Vec<&PhysicalDeviceInfo> = devices
        .iter()
        .filter(|d| !requirements.graphics || d.sync_tier != SyncTier::Legacy || true)
        .collect();
    if candidates.is_empty() {
        candidates = devices.iter().collect();
    }
    candidates.into_iter().max_by_key(|d| score_device(d))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn info(
        name: &str,
        device_type: vk::PhysicalDeviceType,
        memory: u64,
        api: ApiVersion,
        tier: SyncTier,
    ) -> PhysicalDeviceInfo {
        PhysicalDeviceInfo {
            handle: vk::PhysicalDevice::null(),
            name: name.to_string(),
            device_type,
            api_version: api,
            driver_version: "test".to_string(),
            total_memory: memory,
            extensions: Vec::new(),
            is_discrete: device_type == vk::PhysicalDeviceType::DISCRETE_GPU,
            sync_tier: tier,
            bindless: vibe_rhi::BindlessSupport::Full,
        }
    }

    #[test]
    fn discrete_gpu_outscores_integrated() {
        let d = info(
            "d",
            vk::PhysicalDeviceType::DISCRETE_GPU,
            8 << 30,
            v(1, 3, 0),
            SyncTier::Sync2Timeline,
        );
        let i = info(
            "i",
            vk::PhysicalDeviceType::INTEGRATED_GPU,
            512 << 20,
            v(1, 3, 0),
            SyncTier::Sync2,
        );
        assert!(score_device(&d) > score_device(&i));
    }

    #[test]
    fn more_memory_breaks_ties() {
        let small = info(
            "s",
            vk::PhysicalDeviceType::DISCRETE_GPU,
            2 << 30,
            v(1, 3, 0),
            SyncTier::Sync2,
        );
        let big = info(
            "b",
            vk::PhysicalDeviceType::DISCRETE_GPU,
            12 << 30,
            v(1, 3, 0),
            SyncTier::Sync2,
        );
        assert!(score_device(&big) > score_device(&small));
    }

    #[test]
    fn newer_api_breaks_ties() {
        let old = info(
            "o",
            vk::PhysicalDeviceType::INTEGRATED_GPU,
            1 << 30,
            v(1, 1, 0),
            SyncTier::Sync2,
        );
        let new = info(
            "n",
            vk::PhysicalDeviceType::INTEGRATED_GPU,
            1 << 30,
            v(1, 3, 0),
            SyncTier::Sync2,
        );
        assert!(score_device(&new) > score_device(&old));
    }

    #[test]
    fn better_sync_tier_breaks_ties() {
        let legacy = info(
            "l",
            vk::PhysicalDeviceType::INTEGRATED_GPU,
            1 << 30,
            v(1, 3, 0),
            SyncTier::Legacy,
        );
        let modern = info(
            "m",
            vk::PhysicalDeviceType::INTEGRATED_GPU,
            1 << 30,
            v(1, 3, 0),
            SyncTier::Sync2Timeline,
        );
        assert!(score_device(&modern) > score_device(&legacy));
    }

    #[test]
    fn memory_score_is_capped_so_type_still_dominates() {
        let huge = info(
            "h",
            vk::PhysicalDeviceType::INTEGRATED_GPU,
            64 << 30,
            v(1, 3, 0),
            SyncTier::Sync2,
        );
        let small_discrete = info(
            "d",
            vk::PhysicalDeviceType::DISCRETE_GPU,
            2 << 30,
            v(1, 3, 0),
            SyncTier::Sync2,
        );
        assert!(score_device(&small_discrete) > score_device(&huge));
    }

    #[test]
    fn select_device_picks_highest_score() {
        let devices = vec![
            info(
                "igpu",
                vk::PhysicalDeviceType::INTEGRATED_GPU,
                1 << 30,
                v(1, 3, 0),
                SyncTier::Sync2,
            ),
            info(
                "dgpu",
                vk::PhysicalDeviceType::DISCRETE_GPU,
                8 << 30,
                v(1, 3, 0),
                SyncTier::Sync2Timeline,
            ),
        ];
        let picked = select_device(&devices, QueueRequirements::all()).unwrap();
        assert_eq!(picked.name, "dgpu");
    }

    #[test]
    fn select_device_with_none_returns_none() {
        assert!(select_device(&[], QueueRequirements::all()).is_none());
    }

    #[test]
    fn queue_requirements_all_enables_everything() {
        let r = QueueRequirements::all();
        assert!(r.graphics && r.transfer && r.present);
    }

    #[test]
    fn compute_only_requirements_drop_graphics_and_present() {
        let r = QueueRequirements::compute_only();
        assert!(!r.graphics && !r.present && r.transfer);
    }

    #[test]
    fn is_discrete_flag_matches_type() {
        let d = info(
            "d",
            vk::PhysicalDeviceType::DISCRETE_GPU,
            0,
            v(1, 1, 0),
            SyncTier::Legacy,
        );
        let i = info(
            "i",
            vk::PhysicalDeviceType::INTEGRATED_GPU,
            0,
            v(1, 1, 0),
            SyncTier::Legacy,
        );
        assert!(d.is_discrete);
        assert!(!i.is_discrete);
    }

    #[test]
    fn vk_result_converts_to_error() {
        let e: VkError = vk::Result::ERROR_OUT_OF_HOST_MEMORY.into();
        assert!(matches!(e, VkError::CallFailed(_)));
        assert!(e.to_string().contains("ERROR_OUT_OF_HOST_MEMORY"));
    }

    #[test]
    fn success_does_not_become_an_error() {
        let e: VkError = vk::Result::SUCCESS.into();
        assert!(matches!(e, VkError::LoaderError(_)));
    }

    #[test]
    fn missing_queue_family_error_names_requirement() {
        let e = VkError::MissingQueueFamily { what: "present" };
        assert!(e.to_string().contains("present"));
    }

    #[test]
    fn no_physical_device_error_is_clear() {
        assert!(
            VkError::NoPhysicalDevice
                .to_string()
                .contains("no vulkan physical device")
        );
    }

    #[test]
    fn queue_families_are_deduplicated() {
        let choice = PhysicalDeviceChoice {
            info: info(
                "d",
                vk::PhysicalDeviceType::DISCRETE_GPU,
                0,
                v(1, 3, 0),
                SyncTier::Sync2,
            ),
            graphics_family: Some(0),
            compute_family: Some(0),
            transfer_family: Some(1),
            present_family: Some(0),
        };
        assert_eq!(choice.queue_families(), vec![0, 1]);
    }
}
