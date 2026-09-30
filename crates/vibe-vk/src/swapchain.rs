//! Swapchain creation, resize and per-frame acquisition.
//!
//! Choosing a surface format and present mode is a compatibility decision, not
//! a preference: a surface may report exactly one usable format, and picking
//! the wrong present mode is the usual cause of a swapchain that recreates
//! every frame. Both decisions are pure functions of what the driver reports, so
//! they are tested without a device.

use ash::khr::surface::Instance as SurfaceInstance;
use ash::khr::swapchain::Device as SwapchainDevice;
use ash::vk;
use log::{info, warn};

use crate::context::VkError;
use crate::frame::FrameConfig;

/// How the window's size maps to swapchain images.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResizeMode {
    /// Recreate the swapchain at the new size, then continue.
    Recreate,
    /// Signal the caller that the frame should be skipped this tick.
    ///
    /// Used when the window is zero-sized or minimised, where there is nothing
    /// to present and recreating would fail.
    Skip,
}

/// What `acquire_next_image` should do with the frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AcquireResult {
    /// An image was acquired at this index; render into it.
    Ready {
        /// Index into the swapchain's image array.
        index: u32,
    },
    /// The surface changed underneath us; recreate and draw nothing.
    Recreate,
    /// The surface is out of date but still usable; skip this frame.
    Skip,
    /// Timed out waiting for an image; skip this frame.
    Timeout,
}

/// Pick a surface format from what the surface offers.
///
/// Prefers 8-bit BGRA in sRGB, falling back to the first available format. A
/// surface with no formats at all is reported rather than defaulted, because
/// there is nothing valid to pick.
pub fn choose_surface_format(
    formats: &[vk::SurfaceFormatKHR],
) -> Result<vk::SurfaceFormatKHR, VkError> {
    let Some(first) = formats.first().copied() else {
        return Err(VkError::Swapchain(
            "surface reported no formats".to_string(),
        ));
    };

    for format in formats {
        if format.format == vk::Format::B8G8R8A8_SRGB || format.format == vk::Format::R8G8B8A8_SRGB
        {
            return Ok(*format);
        }
    }
    // No sRGB option: take the driver's first choice rather than failing.
    Ok(first)
}

/// Pick a present mode.
///
/// Prefers `MAILBOX` (no tearing, one queued frame), then `FIFO` (always
/// supported, vsync), and falls back to whatever the surface offers rather than
/// failing, since some drivers report only `IMMEDIATE`.
pub fn choose_present_mode(modes: &[vk::PresentModeKHR]) -> vk::PresentModeKHR {
    const PREFERENCE: [vk::PresentModeKHR; 3] = [
        vk::PresentModeKHR::MAILBOX,
        vk::PresentModeKHR::FIFO,
        vk::PresentModeKHR::FIFO_RELAXED,
    ];
    for mode in PREFERENCE {
        if modes.contains(&mode) {
            return mode;
        }
    }
    modes.first().copied().unwrap_or(vk::PresentModeKHR::FIFO)
}

/// The number of swapchain images to request.
///
/// `MIN_IMAGE_COUNT` is the floor the spec guarantees; one more is the usual
/// choice because it lets the GPU work on a frame while the CPU records the
/// next. Never more than the surface's maximum.
pub fn choose_image_count(capabilities: &vk::SurfaceCapabilitiesKHR, desired: u32) -> u32 {
    let min = capabilities.min_image_count.max(1);
    let max = capabilities.max_image_count;
    let wanted = desired.max(min);
    if max == 0 {
        // A maximum of zero means the surface reports no limit.
        return wanted;
    }
    wanted.min(max)
}

/// Whether a resize means recreate or skip.
///
/// A zero extent is a minimised window: there is no image to present, and
/// asking the driver for a zero-sized swapchain is an error.
pub fn classify_extent(
    current: vk::Extent2D,
    capabilities: &vk::SurfaceCapabilitiesKHR,
    desired: (u32, u32),
) -> ResizeMode {
    if desired.0 == 0 || desired.1 == 0 {
        return ResizeMode::Skip;
    }
    if capabilities.current_extent.width != u32::MAX
        && capabilities.current_extent.height != u32::MAX
    {
        // The surface is fixed-size: use its extent and never resize.
        return ResizeMode::Skip;
    }
    if current.width == desired.0 && current.height == desired.1 {
        return ResizeMode::Skip;
    }
    ResizeMode::Recreate
}

/// Clamp a desired extent to the surface's allowed range.
pub fn clamp_extent(
    desired: (u32, u32),
    capabilities: &vk::SurfaceCapabilitiesKHR,
) -> vk::Extent2D {
    vk::Extent2D {
        width: desired.0.clamp(
            capabilities.min_image_extent.width,
            capabilities.max_image_extent.width,
        ),
        height: desired.1.clamp(
            capabilities.min_image_extent.height,
            capabilities.max_image_extent.height,
        ),
    }
}

/// A wrapper around a live swapchain and the decisions behind it.
#[derive(Debug)]
pub struct Swapchain {
    handle: vk::SwapchainKHR,
    format: vk::SurfaceFormatKHR,
    present_mode: vk::PresentModeKHR,
    extent: vk::Extent2D,
    image_count: u32,
    images: Vec<vk::Image>,
}

impl Swapchain {
    /// Create a swapchain for a surface.
    ///
    /// # Safety
    ///
    /// `device` must be a live logical device, `surface_loader` and
    /// `device_loader` must be loaded for the same instance, and `surface` and
    /// `physical_device` must belong to it.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn new(
        swapchain_loader: &SwapchainDevice,
        surface_loader: &SurfaceInstance,
        physical_device: vk::PhysicalDevice,
        surface: vk::SurfaceKHR,
        graphics_queue: u32,
        present_queue: u32,
        desired_extent: (u32, u32),
        old_swapchain: vk::SwapchainKHR,
        config: FrameConfig,
    ) -> Result<Swapchain, VkError> {
        let caps = unsafe {
            surface_loader.get_physical_device_surface_capabilities(physical_device, surface)
        }?;
        let formats = unsafe {
            surface_loader.get_physical_device_surface_formats(physical_device, surface)
        }?;
        let present_modes = unsafe {
            surface_loader.get_physical_device_surface_present_modes(physical_device, surface)
        }?;

        let format = choose_surface_format(&formats)?;
        let present_mode = choose_present_mode(&present_modes);
        let extent = clamp_extent(desired_extent, &caps);
        let image_count = choose_image_count(&caps, config.frames_in_flight as u32 + 1);

        // A composite alpha of OPAQUE is required for most compositors, and
        // anything else is only legal when the surface supports it.
        let composite_alpha = if caps
            .supported_composite_alpha
            .contains(vk::CompositeAlphaFlagsKHR::OPAQUE)
        {
            vk::CompositeAlphaFlagsKHR::OPAQUE
        } else if caps
            .supported_composite_alpha
            .contains(vk::CompositeAlphaFlagsKHR::PRE_MULTIPLIED)
        {
            vk::CompositeAlphaFlagsKHR::PRE_MULTIPLIED
        } else {
            vk::CompositeAlphaFlagsKHR::INHERIT
        };

        // The readback path copies a presented image into a host buffer, which
        // needs TRANSFER_SRC. Only request it when the surface supports it:
        // asking for a usage the surface does not offer makes
        // vkCreateSwapchainKHR fail, which is worse than not being able to
        // read back.
        let mut image_usage = vk::ImageUsageFlags::COLOR_ATTACHMENT;
        if caps
            .supported_usage_flags
            .contains(vk::ImageUsageFlags::TRANSFER_DST)
        {
            image_usage |= vk::ImageUsageFlags::TRANSFER_DST;
        }
        if caps
            .supported_usage_flags
            .contains(vk::ImageUsageFlags::TRANSFER_SRC)
        {
            image_usage |= vk::ImageUsageFlags::TRANSFER_SRC;
        } else {
            warn!(
                "surface does not support TRANSFER_SRC; frames cannot be read back \
                 for verification"
            );
        }

        let queue_families: [u32; 2] = [graphics_queue, present_queue];
        let unique_families = queue_families.len() == 2 && queue_families[0] != queue_families[1];

        let sharing_mode = if unique_families {
            vk::SharingMode::CONCURRENT
        } else {
            vk::SharingMode::EXCLUSIVE
        };

        let create_info = vk::SwapchainCreateInfoKHR {
            surface,
            min_image_count: image_count,
            image_format: format.format,
            image_color_space: format.color_space,
            image_extent: extent,
            image_array_layers: 1,
            image_usage,
            image_sharing_mode: sharing_mode,
            queue_family_index_count: if unique_families { 2 } else { 0 } as u32,
            p_queue_family_indices: if unique_families {
                queue_families.as_ptr()
            } else {
                std::ptr::null()
            },
            pre_transform: caps.current_transform,
            composite_alpha,
            present_mode,
            clipped: vk::TRUE,
            old_swapchain,
            ..Default::default()
        };

        let handle = unsafe { swapchain_loader.create_swapchain(&create_info, None)? };
        let images = unsafe { swapchain_loader.get_swapchain_images(handle)? };

        info!(
            "swapchain created: {}x{} format {:?} present {:?} images {}",
            extent.width,
            extent.height,
            format.format,
            present_mode,
            images.len()
        );

        Ok(Swapchain {
            handle,
            format,
            present_mode,
            extent,
            image_count,
            images,
        })
    }

    /// The swapchain handle.
    pub fn handle(&self) -> vk::SwapchainKHR {
        self.handle
    }

    /// The chosen surface format.
    pub fn format(&self) -> vk::SurfaceFormatKHR {
        self.format
    }

    /// The chosen present mode.
    pub fn present_mode(&self) -> vk::PresentModeKHR {
        self.present_mode
    }

    /// The swapchain's extent in pixels.
    pub fn extent(&self) -> vk::Extent2D {
        self.extent
    }

    /// The number of images requested.
    pub fn image_count(&self) -> u32 {
        self.image_count
    }

    /// The swapchain's images.
    pub fn images(&self) -> &[vk::Image] {
        &self.images
    }

    /// Try to acquire the next image to render into.
    ///
    /// # Safety
    ///
    /// `semaphore` and `fence` must be valid and unsignalled, or null.
    pub unsafe fn acquire(
        &self,
        swapchain_loader: &SwapchainDevice,
        timeout_nanos: u64,
        semaphore: vk::Semaphore,
        fence: vk::Fence,
    ) -> Result<AcquireResult, VkError> {
        // ash returns (index, suboptimal): the index is always valid, and
        // `suboptimal` means the surface changed but the image is usable.
        let (index, suboptimal) = unsafe {
            swapchain_loader.acquire_next_image(self.handle, timeout_nanos, semaphore, fence)
        }?;
        if suboptimal {
            return Ok(AcquireResult::Recreate);
        }
        Ok(AcquireResult::Ready { index })
    }

    /// Present an image.
    ///
    /// # Safety
    ///
    /// `semaphore` must be the one signalled by the submit that rendered
    /// `image_index`.
    pub unsafe fn present(
        &self,
        swapchain_loader: &SwapchainDevice,
        queue: vk::Queue,
        image_index: u32,
        _semaphore: vk::Semaphore,
    ) -> Result<AcquireResult, VkError> {
        let info = vk::PresentInfoKHR {
            swapchain_count: 1,
            p_swapchains: &self.handle,
            p_image_indices: &image_index,
            p_results: std::ptr::null_mut(),
            ..Default::default()
        };
        let result = unsafe { swapchain_loader.queue_present(queue, &info) };
        match result {
            Ok(suboptimal) if suboptimal => Ok(AcquireResult::Recreate),
            Ok(_) => Ok(AcquireResult::Ready { index: image_index }),
            Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => Ok(AcquireResult::Recreate),
            Err(e) => Err(VkError::from(e)),
        }
    }

    /// Destroy the swapchain.
    ///
    /// # Safety
    ///
    /// No image from this swapchain may still be in use by the GPU.
    pub unsafe fn destroy(&mut self, swapchain_loader: &SwapchainDevice) {
        unsafe {
            swapchain_loader.destroy_swapchain(self.handle, None);
        }
        self.handle = vk::SwapchainKHR::null();
        self.images.clear();
    }

    /// Warn when the driver reports something worth knowing.
    pub fn report_quirks(&self) {
        if self.image_count < 2 {
            warn!(
                "swapchain has {} image(s); the GPU may stall waiting for the CPU",
                self.image_count
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn format(f: vk::Format) -> vk::SurfaceFormatKHR {
        vk::SurfaceFormatKHR {
            format: f,
            color_space: vk::ColorSpaceKHR::SRGB_NONLINEAR,
        }
    }

    fn caps() -> vk::SurfaceCapabilitiesKHR {
        vk::SurfaceCapabilitiesKHR {
            min_image_count: 2,
            max_image_count: 8,
            // u32::MAX on both axes is the spec's "resizable" sentinel.
            current_extent: vk::Extent2D {
                width: u32::MAX,
                height: u32::MAX,
            },
            min_image_extent: vk::Extent2D {
                width: 1,
                height: 1,
            },
            max_image_extent: vk::Extent2D {
                width: 4096,
                height: 4096,
            },
            max_image_array_layers: 1,
            supported_transforms: vk::SurfaceTransformFlagsKHR::IDENTITY,
            current_transform: vk::SurfaceTransformFlagsKHR::IDENTITY,
            supported_composite_alpha: vk::CompositeAlphaFlagsKHR::OPAQUE,
            supported_usage_flags: vk::ImageUsageFlags::COLOR_ATTACHMENT,
            ..Default::default()
        }
    }

    #[test]
    fn bgra_srgb_is_preferred() {
        let formats = [
            format(vk::Format::R8G8B8A8_UNORM),
            format(vk::Format::B8G8R8A8_SRGB),
        ];
        let picked = choose_surface_format(&formats).unwrap();
        assert_eq!(picked.format, vk::Format::B8G8R8A8_SRGB);
    }

    #[test]
    fn rgba_srgb_is_also_acceptable() {
        let formats = [
            format(vk::Format::B8G8R8A8_UNORM),
            format(vk::Format::R8G8B8A8_SRGB),
        ];
        let picked = choose_surface_format(&formats).unwrap();
        assert_eq!(picked.format, vk::Format::R8G8B8A8_SRGB);
    }

    #[test]
    fn falls_back_to_the_first_format_without_srgb() {
        let formats = [
            format(vk::Format::R8G8B8A8_UNORM),
            format(vk::Format::B8G8R8A8_UNORM),
        ];
        let picked = choose_surface_format(&formats).unwrap();
        assert_eq!(picked.format, vk::Format::R8G8B8A8_UNORM);
    }

    #[test]
    fn no_formats_is_an_error() {
        let err = choose_surface_format(&[]).unwrap_err();
        assert!(matches!(err, VkError::Swapchain(_)));
        assert!(err.to_string().contains("no formats"));
    }

    #[test]
    fn mailbox_is_preferred() {
        let modes = [
            vk::PresentModeKHR::IMMEDIATE,
            vk::PresentModeKHR::FIFO,
            vk::PresentModeKHR::MAILBOX,
        ];
        assert_eq!(choose_present_mode(&modes), vk::PresentModeKHR::MAILBOX);
    }

    #[test]
    fn fifo_is_used_when_mailbox_is_absent() {
        let modes = [vk::PresentModeKHR::IMMEDIATE, vk::PresentModeKHR::FIFO];
        assert_eq!(choose_present_mode(&modes), vk::PresentModeKHR::FIFO);
    }

    #[test]
    fn fifo_relaxed_is_used_before_immediate() {
        let modes = [
            vk::PresentModeKHR::IMMEDIATE,
            vk::PresentModeKHR::FIFO_RELAXED,
        ];
        assert_eq!(
            choose_present_mode(&modes),
            vk::PresentModeKHR::FIFO_RELAXED
        );
    }

    #[test]
    fn immediate_only_surface_still_works() {
        let modes = [vk::PresentModeKHR::IMMEDIATE];
        assert_eq!(choose_present_mode(&modes), vk::PresentModeKHR::IMMEDIATE);
    }

    #[test]
    fn no_present_modes_defaults_to_fifo() {
        assert_eq!(choose_present_mode(&[]), vk::PresentModeKHR::FIFO);
    }

    #[test]
    fn image_count_respects_the_minimum() {
        let c = caps();
        assert_eq!(
            choose_image_count(&c, 1),
            2,
            "one is below the surface minimum"
        );
    }

    #[test]
    fn image_count_respects_the_maximum() {
        let c = caps();
        assert_eq!(choose_image_count(&c, 100), 8);
    }

    #[test]
    fn image_count_passes_a_reasonable_request_through() {
        let c = caps();
        assert_eq!(choose_image_count(&c, 3), 3);
    }

    #[test]
    fn image_count_handles_an_unlimited_maximum() {
        let c = vk::SurfaceCapabilitiesKHR {
            max_image_count: 0,
            ..caps()
        };
        assert_eq!(choose_image_count(&c, 4), 4);
    }

    #[test]
    fn zero_extent_means_skip() {
        assert_eq!(
            classify_extent(
                vk::Extent2D {
                    width: 800,
                    height: 600
                },
                &caps(),
                (0, 600)
            ),
            ResizeMode::Skip
        );
    }

    #[test]
    fn unchanged_extent_means_skip() {
        let current = vk::Extent2D {
            width: 800,
            height: 600,
        };
        assert_eq!(
            classify_extent(current, &caps(), (800, 600)),
            ResizeMode::Skip
        );
    }

    #[test]
    fn changed_extent_means_recreate() {
        let current = vk::Extent2D {
            width: 800,
            height: 600,
        };
        assert_eq!(
            classify_extent(current, &caps(), (1024, 768)),
            ResizeMode::Recreate
        );
    }

    #[test]
    fn a_fixed_surface_never_recreates() {
        let c = vk::SurfaceCapabilitiesKHR {
            current_extent: vk::Extent2D {
                width: 800,
                height: 600,
            },
            ..caps()
        };
        let mode = classify_extent(
            vk::Extent2D {
                width: 800,
                height: 600,
            },
            &c,
            (1920, 1080),
        );
        assert_eq!(mode, ResizeMode::Skip, "a fixed surface cannot be resized");
    }

    #[test]
    fn clamping_applies_the_surface_minimum() {
        let c = vk::SurfaceCapabilitiesKHR {
            min_image_extent: vk::Extent2D {
                width: 640,
                height: 480,
            },
            ..caps()
        };
        let e = clamp_extent((100, 100), &c);
        assert_eq!((e.width, e.height), (640, 480));
    }

    #[test]
    fn clamping_applies_the_surface_maximum() {
        let e = clamp_extent((99999, 99999), &caps());
        assert_eq!((e.width, e.height), (4096, 4096));
    }

    #[test]
    fn clamping_passes_a_valid_extent_through() {
        let e = clamp_extent((800, 600), &caps());
        assert_eq!((e.width, e.height), (800, 600));
    }

    #[test]
    fn acquire_result_variants_are_distinct() {
        assert_ne!(AcquireResult::Ready { index: 0 }, AcquireResult::Recreate);
        assert_ne!(AcquireResult::Recreate, AcquireResult::Skip);
        assert_ne!(AcquireResult::Skip, AcquireResult::Timeout);
    }

    #[test]
    fn resize_mode_variants_are_distinct() {
        assert_ne!(ResizeMode::Recreate, ResizeMode::Skip);
    }

    #[test]
    fn a_two_frame_config_asks_for_three_images() {
        // One more than frames-in-flight is what lets the GPU stay busy.
        let config = FrameConfig::with_frames(crate::sync::SyncTier::Sync2, 2);
        assert_eq!(
            choose_image_count(&caps(), config.frames_in_flight as u32 + 1),
            3
        );
    }
}
