//! Sync tier detection: Legacy, Sync2, or Sync2Timeline.

/// How the engine will emit synchronisation for this device.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum SyncTier {
    /// Vulkan 1.1: `vkCmdPipelineBarrier` only. No timeline semaphores, no
    /// `vkCmdPipelineBarrier2`.
    Legacy,
    /// `VK_KHR_synchronization2`: `vkCmdPipelineBarrier2` available, but queue
    /// waits still need binary semaphores or fences.
    Sync2,
    /// `VK_KHR_synchronization2` plus `VK_KHR_timeline_semaphore`: per-queue
    /// timeline semaphores, which is what frame pacing needs.
    Sync2Timeline,
}

impl SyncTier {
    /// The name used in logs and the editor's device panel.
    pub const fn name(self) -> &'static str {
        match self {
            SyncTier::Legacy => "Legacy",
            SyncTier::Sync2 => "Sync2",
            SyncTier::Sync2Timeline => "Sync2Timeline",
        }
    }

    /// True when `vkCmdPipelineBarrier2` may be called.
    pub const fn supports_barrier2(self) -> bool {
        matches!(self, SyncTier::Sync2 | SyncTier::Sync2Timeline)
    }

    /// True when timeline semaphores are available.
    pub const fn supports_timeline(self) -> bool {
        matches!(self, SyncTier::Sync2Timeline)
    }
}

impl std::fmt::Display for SyncTier {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

/// A Vulkan API version packed into one integer.
pub type ApiVersion = u32;

/// Build a Vulkan API version.
///
/// `variant` is the high bits, which only a vendor-provisional extension
/// would set; the engine always passes 0.
pub const fn api_version(variant: u32, major: u32, minor: u32, patch: u32) -> ApiVersion {
    (variant << 29) | (major << 22) | (minor << 12) | patch
}

/// The major component of a version.
pub const fn major(v: ApiVersion) -> u32 {
    (v >> 22) & 0x7f
}

/// The minor component of a version.
pub const fn minor(v: ApiVersion) -> u32 {
    (v >> 12) & 0x3ff
}

/// The patch component of a version.
pub const fn patch(v: ApiVersion) -> u32 {
    v & 0xfff
}

/// Format a version as `major.minor.patch`.
pub fn version_string(v: ApiVersion) -> String {
    format!("{}.{}.{}", major(v), minor(v), patch(v))
}

/// One Vulkan extension the backend knows about.
pub struct ExtensionInfo {
    /// `VK_KHR_*` / `VK_EXT_*` name.
    pub name: &'static str,
    /// The Vulkan version that promoted it to core, if any.
    pub promoted_in: Option<ApiVersion>,
    /// True when the backend will try to enable it if the device reports it.
    pub used: bool,
}

const fn ext(name: &'static str, promoted_in: Option<ApiVersion>, used: bool) -> ExtensionInfo {
    ExtensionInfo {
        name,
        promoted_in,
        used,
    }
}

/// The extension table.
///
/// Listing an extension here states "we know how to use it", not "we turned it
/// on"; the `used` flag says which are auto-enabled when the device reports
/// them. After promotion an extension need not be enumerated at all, which is
/// why the version check runs alongside the name check.
pub const DECLARED_EXTENSIONS: &[ExtensionInfo] = &[
    // Core in 1.1
    ext(
        "VK_KHR_get_physical_device_properties2",
        Some(api_version(0, 1, 1, 0)),
        false,
    ),
    ext(
        "VK_KHR_device_group_creation",
        Some(api_version(0, 1, 1, 0)),
        false,
    ),
    ext(
        "VK_KHR_external_memory_capabilities",
        Some(api_version(0, 1, 1, 0)),
        false,
    ),
    ext(
        "VK_KHR_external_semaphore_capabilities",
        Some(api_version(0, 1, 1, 0)),
        false,
    ),
    ext(
        "VK_KHR_external_fence_capabilities",
        Some(api_version(0, 1, 1, 0)),
        false,
    ),
    ext("VK_KHR_maintenance1", None, false),
    ext(
        "VK_KHR_variable_pointers",
        Some(api_version(0, 1, 1, 0)),
        false,
    ),
    // Core in 1.2
    ext(
        "VK_KHR_driver_properties",
        Some(api_version(0, 1, 2, 0)),
        false,
    ),
    ext(
        "VK_KHR_buffer_device_address",
        Some(api_version(0, 1, 2, 0)),
        true,
    ),
    ext(
        "VK_KHR_descriptor_update_template",
        Some(api_version(0, 1, 2, 0)),
        false,
    ),
    ext(
        "VK_KHR_sampler_mirror_clamp_to_edge",
        Some(api_version(0, 1, 2, 0)),
        false,
    ),
    ext(
        "VK_KHR_shader_draw_parameters",
        Some(api_version(0, 1, 2, 0)),
        false,
    ),
    ext(
        "VK_KHR_subgroup_uniform_control_flow",
        Some(api_version(0, 1, 2, 0)),
        false,
    ),
    ext("VK_KHR_8bit_storage", Some(api_version(0, 1, 2, 0)), false),
    ext("VK_KHR_16bit_storage", Some(api_version(0, 1, 2, 0)), false),
    ext(
        "VK_KHR_sampler_ycbcr_conversion",
        Some(api_version(0, 1, 2, 0)),
        false,
    ),
    // Core in 1.3
    ext(
        "VK_KHR_dynamic_rendering",
        Some(api_version(0, 1, 3, 0)),
        true,
    ),
    ext(
        "VK_KHR_synchronization2",
        Some(api_version(0, 1, 3, 0)),
        true,
    ),
    ext("VK_KHR_copy_commands2", Some(api_version(0, 1, 3, 0)), true),
    ext(
        "VK_KHR_format_feature_flags2",
        Some(api_version(0, 1, 3, 0)),
        false,
    ),
    ext("VK_KHR_maintenance3", Some(api_version(0, 1, 3, 0)), false),
    ext("VK_KHR_maintenance4", Some(api_version(0, 1, 3, 0)), false),
    ext(
        "VK_KHR_shader_non_semantic_info",
        Some(api_version(0, 1, 3, 0)),
        false,
    ),
    ext(
        "VK_KHR_zero_initialize_workgroup_memory",
        Some(api_version(0, 1, 3, 0)),
        false,
    ),
    // Never promoted; required by the feature set.
    ext("VK_KHR_timeline_semaphore", None, true),
    ext("VK_KHR_swapchain", None, true),
    ext("VK_KHR_bindless_texture", None, true),
    ext("VK_EXT_descriptor_indexing", None, true),
    ext("VK_KHR_surface", None, true),
    // Optional, enabled when the device reports them.
    ext("VK_EXT_debug_utils", None, false),
    ext("VK_EXT_present_wait", None, false),
    ext("VK_EXT_swapchain_maintenance1", None, false),
    ext("VK_EXT_full_screen_exclusive", None, false),
    ext("VK_EXT_private_data", None, false),
    ext("VK_EXT_debug_marker", None, false),
    ext("VK_EXT_calibrated_timestamps", None, false),
    ext("VK_EXT_headless_surface", None, false),
    ext("VK_EXT_vertex_attribute_divisor", None, false),
    ext("VK_KHR_deferred_host_operations", None, false),
    ext("VK_KHR_pipeline_library", None, false),
    ext("VK_KHR_portability_subset", None, false),
    ext("VK_KHR_portability_enumeration", None, false),
    ext("VK_KHR_external_memory_fd", None, false),
    ext("VK_KHR_external_semaphore_fd", None, false),
    ext("VK_EXT_memory_budget", None, false),
    ext("VK_KHR_acceleration_structure", None, false),
    ext("VK_KHR_ray_tracing_pipeline", None, false),
    ext("VK_EXT_mesh_shader", None, false),
    ext("VK_EXT_shader_atomic_float", None, false),
    // Core in 1.4
    ext(
        "VK_KHR_dynamic_rendering_local_read",
        Some(api_version(0, 1, 4, 0)),
        false,
    ),
    ext(
        "VK_KHR_global_priority",
        Some(api_version(0, 1, 4, 0)),
        false,
    ),
    ext(
        "VK_KHR_shader_quad_control",
        Some(api_version(0, 1, 4, 0)),
        false,
    ),
    ext(
        "VK_KHR_sampler_filter_minmax",
        Some(api_version(0, 1, 4, 0)),
        false,
    ),
    ext(
        "VK_KHR_shader_float_controls2",
        Some(api_version(0, 1, 4, 0)),
        false,
    ),
    ext(
        "VK_KHR_vertex_attribute_divisor",
        Some(api_version(0, 1, 4, 0)),
        false,
    ),
    ext(
        "VK_KHR_index_type_uint8",
        Some(api_version(0, 1, 4, 0)),
        false,
    ),
    ext("VK_KHR_map_memory2", Some(api_version(0, 1, 4, 0)), false),
    ext("VK_KHR_maintenance5", Some(api_version(0, 1, 4, 0)), false),
    ext("VK_EXT_multiview", Some(api_version(0, 1, 3, 0)), false),
    ext(
        "VK_EXT_shader_replicated_composites",
        Some(api_version(0, 1, 3, 0)),
        false,
    ),
    ext(
        "VK_EXT_scalar_block_layout",
        Some(api_version(0, 1, 2, 0)),
        false,
    ),
    ext(
        "VK_EXT_separate_stencil_use_ref_layout",
        Some(api_version(0, 1, 2, 0)),
        false,
    ),
    ext(
        "VK_KHR_uniform_buffer_standard_layout",
        Some(api_version(0, 1, 2, 0)),
        false,
    ),
    ext(
        "VK_KHR_sampler_filter_minmax_properties",
        Some(api_version(0, 1, 2, 0)),
        false,
    ),
    ext(
        "VK_KHR_storage_buffer_storage_class",
        Some(api_version(0, 1, 2, 0)),
        false,
    ),
];

/// True when `name` is one the backend knows how to use.
pub fn is_declared(name: &str) -> bool {
    DECLARED_EXTENSIONS.iter().any(|e| e.name == name)
}

/// True when `name` is declared and the backend will try to enable it.
pub fn is_wanted(name: &str) -> bool {
    DECLARED_EXTENSIONS.iter().any(|e| e.name == name && e.used)
}

/// The core API version an extension was promoted in, if any.
pub fn promoted_in(name: &str) -> Option<ApiVersion> {
    DECLARED_EXTENSIONS
        .iter()
        .find(|e| e.name == name)
        .and_then(|e| e.promoted_in)
}

/// Choose the sync tier from the device's API version and extension list.
///
/// `extensions` must be device extension names, not instance ones.
pub fn detect_sync_tier(api: ApiVersion, extensions: &[&str]) -> SyncTier {
    let has = |n: &str| extensions.iter().any(|e| *e == n);
    let sync2 = has("VK_KHR_synchronization2") || api >= api_version(0, 1, 3, 0);
    if !sync2 {
        return SyncTier::Legacy;
    }
    if has("VK_KHR_timeline_semaphore") || api >= api_version(0, 1, 2, 0) {
        SyncTier::Sync2Timeline
    } else {
        SyncTier::Sync2
    }
}

/// Classify how a device can address resources from shaders.
///
/// Bindless needs both `VK_EXT_descriptor_indexing` and
/// `VK_KHR_bindless_texture`; the iGPU path on Intel's Tiger Lake exposes only
/// the former, so it lands in [`BindlessSupport::Partial`].
pub fn detect_bindless_support(extensions: &[&str]) -> vibe_rhi::BindlessSupport {
    let has = |n: &str| extensions.iter().any(|e| *e == n);
    if has("VK_KHR_bindless_texture") && has("VK_EXT_descriptor_indexing") {
        return vibe_rhi::BindlessSupport::Full;
    }
    if has("VK_EXT_descriptor_indexing") {
        return vibe_rhi::BindlessSupport::Partial {
            max_indexed_descriptors: 4096,
        };
    }
    vibe_rhi::BindlessSupport::None
}

/// The tier implied by the loader version.
///
/// A 1.0 loader cannot expose 1.1 core entry points, so the tier is capped
/// even when the device itself is newer.
pub fn tier_for_loader(loader: ApiVersion, api: ApiVersion, extensions: &[&str]) -> SyncTier {
    if loader < api_version(0, 1, 1, 0) {
        return SyncTier::Legacy;
    }
    detect_sync_tier(api, extensions)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v(major: u32, minor: u32, patch: u32) -> ApiVersion {
        api_version(0, major, minor, patch)
    }

    #[test]
    fn version_components_round_trip() {
        let x = v(1, 3, 280);
        assert_eq!((major(x), minor(x), patch(x)), (1, 3, 280));
        assert_eq!(version_string(x), "1.3.280");
    }

    #[test]
    fn variant_bits_do_not_collide_with_major_or_minor() {
        // variant occupies the top 3 bits, so a nonzero variant must not
        // corrupt the major or minor fields.
        let plain = api_version(0, 1, 3, 280);
        let variant = api_version(1, 1, 3, 280);
        assert_eq!(major(variant), major(plain));
        assert_eq!(minor(variant), minor(plain));
        assert_eq!(patch(variant), patch(plain));
        assert_ne!(variant, plain, "the variant must still be visible");
    }

    #[test]
    fn version_string_of_zero() {
        assert_eq!(version_string(0), "0.0.0");
    }

    #[test]
    fn names_are_stable() {
        assert_eq!(SyncTier::Legacy.name(), "Legacy");
        assert_eq!(SyncTier::Sync2.name(), "Sync2");
        assert_eq!(SyncTier::Sync2Timeline.name(), "Sync2Timeline");
        assert_eq!(SyncTier::Sync2.to_string(), "Sync2");
    }

    #[test]
    fn legacy_tier_caps_every_feature() {
        assert!(!SyncTier::Legacy.supports_barrier2());
        assert!(!SyncTier::Legacy.supports_timeline());
    }

    #[test]
    fn sync2_has_barrier2_but_no_timeline() {
        assert!(SyncTier::Sync2.supports_barrier2());
        assert!(!SyncTier::Sync2.supports_timeline());
    }

    #[test]
    fn sync2_timeline_has_both() {
        assert!(SyncTier::Sync2Timeline.supports_barrier2());
        assert!(SyncTier::Sync2Timeline.supports_timeline());
    }

    #[test]
    fn tier_ordering_is_ascending_capability() {
        assert!(SyncTier::Legacy < SyncTier::Sync2);
        assert!(SyncTier::Sync2 < SyncTier::Sync2Timeline);
    }

    #[test]
    fn vulkan_1_0_device_with_no_extensions_is_legacy() {
        assert_eq!(detect_sync_tier(v(1, 0, 0), &[]), SyncTier::Legacy);
    }

    #[test]
    fn vulkan_1_1_device_without_synchronization2_is_legacy() {
        assert_eq!(detect_sync_tier(v(1, 1, 0), &[]), SyncTier::Legacy);
    }

    #[test]
    fn synchronization2_extension_alone_gives_sync2() {
        assert_eq!(
            detect_sync_tier(v(1, 1, 0), &["VK_KHR_synchronization2"]),
            SyncTier::Sync2
        );
    }

    #[test]
    fn both_extensions_give_sync2_timeline() {
        assert_eq!(
            detect_sync_tier(
                v(1, 1, 0),
                &["VK_KHR_synchronization2", "VK_KHR_timeline_semaphore"]
            ),
            SyncTier::Sync2Timeline
        );
    }

    #[test]
    fn core_1_3_promotes_synchronization2() {
        assert_eq!(detect_sync_tier(v(1, 3, 0), &[]), SyncTier::Sync2Timeline);
    }

    #[test]
    fn core_1_2_promotes_timeline_semaphore() {
        assert_eq!(
            detect_sync_tier(v(1, 2, 0), &["VK_KHR_synchronization2"]),
            SyncTier::Sync2Timeline
        );
    }

    #[test]
    fn core_1_1_with_synchronization2_gives_sync2() {
        assert_eq!(
            detect_sync_tier(v(1, 1, 0), &["VK_KHR_synchronization2"]),
            SyncTier::Sync2
        );
    }

    #[test]
    fn unrelated_extensions_do_not_upgrade_tier() {
        assert_eq!(
            detect_sync_tier(v(1, 1, 0), &["VK_KHR_surface", "VK_KHR_swapchain"]),
            SyncTier::Legacy
        );
    }

    #[test]
    fn bindless_needs_both_extensions() {
        assert_eq!(
            detect_bindless_support(&["VK_KHR_bindless_texture", "VK_EXT_descriptor_indexing"]),
            vibe_rhi::BindlessSupport::Full
        );
    }

    #[test]
    fn descriptor_indexing_alone_is_partial() {
        // This is the Tiger Lake iGPU case.
        assert_eq!(
            detect_bindless_support(&["VK_EXT_descriptor_indexing", "VK_KHR_swapchain"]),
            vibe_rhi::BindlessSupport::Partial {
                max_indexed_descriptors: 4096
            }
        );
    }

    #[test]
    fn bindless_texture_alone_is_not_enough() {
        assert_eq!(
            detect_bindless_support(&["VK_KHR_bindless_texture"]),
            vibe_rhi::BindlessSupport::None
        );
    }

    #[test]
    fn no_extensions_means_no_bindless() {
        assert_eq!(
            detect_bindless_support(&[]),
            vibe_rhi::BindlessSupport::None
        );
    }

    #[test]
    fn old_loader_caps_the_tier() {
        assert_eq!(
            tier_for_loader(v(1, 0, 0), v(1, 3, 0), &[]),
            SyncTier::Legacy
        );
    }

    #[test]
    fn modern_loader_defers_to_detection() {
        assert_eq!(
            tier_for_loader(v(1, 3, 0), v(1, 3, 0), &[]),
            SyncTier::Sync2Timeline
        );
    }

    #[test]
    fn extension_table_is_declared_and_queryable() {
        assert!(is_declared("VK_KHR_synchronization2"));
        assert!(is_declared("VK_KHR_timeline_semaphore"));
        assert!(!is_declared("VK_EXT_not_a_real_extension"));
    }

    #[test]
    fn wantedness_differs_from_declared() {
        assert!(is_wanted("VK_KHR_synchronization2"));
        assert!(is_declared("VK_EXT_debug_utils"));
        assert!(
            !is_wanted("VK_EXT_debug_utils"),
            "optional extension is known but not auto-enabled"
        );
    }

    #[test]
    fn promotion_versions_are_recorded() {
        assert_eq!(promoted_in("VK_KHR_synchronization2"), Some(v(1, 3, 0)));
        assert_eq!(promoted_in("VK_KHR_timeline_semaphore"), None);
        assert_eq!(promoted_in("VK_KHR_dynamic_rendering"), Some(v(1, 3, 0)));
        assert_eq!(promoted_in("VK_EXT_debug_utils"), None);
    }

    #[test]
    fn table_has_no_duplicate_names() {
        let mut names: Vec<&str> = DECLARED_EXTENSIONS.iter().map(|e| e.name).collect();
        let before = names.len();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), before, "duplicate extension in the table");
    }

    #[test]
    fn table_covers_sixty_extensions() {
        assert!(
            DECLARED_EXTENSIONS.len() >= 60,
            "got {}",
            DECLARED_EXTENSIONS.len()
        );
    }

    #[test]
    fn every_promotion_version_is_a_real_vulkan_release() {
        for e in DECLARED_EXTENSIONS {
            if let Some(v) = e.promoted_in {
                assert!(
                    minor(v) <= 4,
                    "{} claims promotion in 1.{}",
                    e.name,
                    minor(v)
                );
            }
        }
    }

    #[test]
    fn every_promoted_version_is_above_1_0() {
        for e in DECLARED_EXTENSIONS {
            if let Some(v) = e.promoted_in {
                assert!(
                    v >= api_version(0, 1, 1, 0),
                    "{} claims promotion below 1.1",
                    e.name
                );
            }
        }
    }

    #[test]
    fn feature_set_is_declared_as_wanted() {
        for required in [
            "VK_KHR_dynamic_rendering",
            "VK_KHR_synchronization2",
            "VK_KHR_timeline_semaphore",
            "VK_KHR_swapchain",
            "VK_KHR_bindless_texture",
            "VK_EXT_descriptor_indexing",
            "VK_KHR_buffer_device_address",
        ] {
            assert!(
                is_wanted(required),
                "{required} must be auto-enabled when present"
            );
        }
    }
}
