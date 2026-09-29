//! Creating a Vulkan surface for a window.
//!
//! The surface is platform-specific: Wayland on niri, XCB or XLIB on X11.
//! Which extension to use is decided here, so `create_instance` can be asked
//! for exactly the right one and nothing else.

use ash::khr;
use ash::vk;
use raw_window_handle::{HasDisplayHandle, HasWindowHandle, RawDisplayHandle, RawWindowHandle};
use winit::window::Window;

use crate::error::WindowError;

/// Which window-system extension a raw handle calls for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SurfaceKind {
    /// `VK_KHR_wayland_surface`.
    Wayland,
    /// `VK_KHR_xcb_surface`.
    Xcb,
    /// `VK_KHR_xlib_surface`.
    Xlib,
}

impl SurfaceKind {
    /// The instance extension name for this platform.
    pub const fn extension(self) -> &'static str {
        match self {
            SurfaceKind::Wayland => "VK_KHR_wayland_surface",
            SurfaceKind::Xcb => "VK_KHR_xcb_surface",
            SurfaceKind::Xlib => "VK_KHR_xlib_surface",
        }
    }

    /// The name used in logs.
    pub const fn name(self) -> &'static str {
        match self {
            SurfaceKind::Wayland => "wayland",
            SurfaceKind::Xcb => "xcb",
            SurfaceKind::Xlib => "xlib",
        }
    }
}

/// What is needed to create a surface.
#[derive(Debug, Clone, Copy)]
pub struct SurfaceDesc<'a> {
    /// The instance extensions to request, including the surface one.
    pub instance_extensions: &'a [*const std::os::raw::c_char],
    /// Which platform extension the window needs.
    pub kind: SurfaceKind,
}

/// Work out which surface extension a window needs.
///
/// # Safety
///
/// The handles must be live; they come from a live window.
pub fn surface_kind(window: &Window) -> Result<SurfaceKind, WindowError> {
    let display = window
        .display_handle()
        .map_err(|_| WindowError::NoHandle("display"))?;
    match display.as_raw() {
        RawDisplayHandle::Wayland(w) => {
            let _ = w;
            Ok(SurfaceKind::Wayland)
        }
        RawDisplayHandle::Xcb(_) => Ok(SurfaceKind::Xcb),
        RawDisplayHandle::Xlib(_) => Ok(SurfaceKind::Xlib),
        other => Err(WindowError::Surface(format!(
            "unsupported display handle {other:?}"
        ))),
    }
}

/// The instance extensions a window's platform needs.
///
/// Always includes `VK_KHR_surface` plus the platform's own extension, since
/// neither is implied.
pub fn required_instance_extensions(
    window: &Window,
) -> Result<Vec<*const std::os::raw::c_char>, WindowError> {
    let kind = surface_kind(window)?;
    Ok(vec![
        c"VK_KHR_surface".as_ptr(),
        match kind {
            SurfaceKind::Wayland => c"VK_KHR_wayland_surface".as_ptr(),
            SurfaceKind::Xcb => c"VK_KHR_xcb_surface".as_ptr(),
            SurfaceKind::Xlib => c"VK_KHR_xlib_surface".as_ptr(),
        },
    ])
}

/// Create a Vulkan surface for a window.
///
/// # Safety
///
/// `entry` and `instance` must be live, and the instance must have been created
/// with the extensions [`required_instance_extensions`] reports.
pub unsafe fn create_surface(
    entry: &ash::Entry,
    instance: &ash::Instance,
    window: &Window,
) -> Result<(vk::SurfaceKHR, SurfaceKind), WindowError> {
    let display = window
        .display_handle()
        .map_err(|_| WindowError::NoHandle("display"))?;
    let handle = window
        .window_handle()
        .map_err(|_| WindowError::NoHandle("window"))?;
    let kind = surface_kind(window)?;

    // Each platform has its own extension loader, built from the same entry
    // and instance; VK_KHR_surface itself carries no creation call.
    let surface = match (display.as_raw(), handle.as_raw()) {
        (RawDisplayHandle::Wayland(d), RawWindowHandle::Wayland(w)) => {
            let loader = khr::wayland_surface::Instance::new(entry, instance);
            let create = vk::WaylandSurfaceCreateInfoKHR {
                display: d.display.as_ptr(),
                surface: w.surface.as_ptr(),
                ..Default::default()
            };
            unsafe { loader.create_wayland_surface(&create, None) }
        }
        (RawDisplayHandle::Xcb(d), RawWindowHandle::Xcb(w)) => {
            // XCB takes no visual id: it is looked up from the window.
            let Some(connection) = d.connection else {
                return Err(WindowError::Surface(
                    "xcb display has no connection".to_string(),
                ));
            };
            let loader = khr::xcb_surface::Instance::new(entry, instance);
            let create = vk::XcbSurfaceCreateInfoKHR {
                connection: connection.as_ptr() as *mut _,
                window: w.window.get(),
                ..Default::default()
            };
            unsafe { loader.create_xcb_surface(&create, None) }
        }
        (RawDisplayHandle::Xlib(d), RawWindowHandle::Xlib(w)) => {
            let Some(display) = d.display else {
                return Err(WindowError::Surface(
                    "xlib display handle is empty".to_string(),
                ));
            };
            let loader = khr::xlib_surface::Instance::new(entry, instance);
            // Like XCB, Xlib takes the visual from the window.
            let _ = d.screen;
            let create = vk::XlibSurfaceCreateInfoKHR {
                dpy: display.as_ptr(),
                window: w.window,
                ..Default::default()
            };
            unsafe { loader.create_xlib_surface(&create, None) }
        }
        (d, w) => {
            return Err(WindowError::Surface(format!(
                "display {d:?} and window {w:?} are from different platforms"
            )));
        }
    };

    match surface {
        Ok(s) => {
            log::info!("vulkan surface created for the {} backend", kind.name());
            Ok((s, kind))
        }
        Err(e) => Err(WindowError::Surface(e.to_string())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wayland_maps_to_its_extension() {
        assert_eq!(SurfaceKind::Wayland.extension(), "VK_KHR_wayland_surface");
        assert_eq!(SurfaceKind::Wayland.name(), "wayland");
    }

    #[test]
    fn x11_maps_to_its_extensions() {
        assert_eq!(SurfaceKind::Xcb.extension(), "VK_KHR_xcb_surface");
        assert_eq!(SurfaceKind::Xlib.extension(), "VK_KHR_xlib_surface");
    }

    #[test]
    fn every_kind_has_a_distinct_extension() {
        let all = [SurfaceKind::Wayland, SurfaceKind::Xcb, SurfaceKind::Xlib];
        for a in all {
            for b in all {
                if a != b {
                    assert_ne!(a.extension(), b.extension(), "{a:?} vs {b:?}");
                }
            }
        }
    }

    #[test]
    fn the_surface_extension_is_never_implied() {
        // VK_KHR_surface must be requested explicitly on every platform, which
        // is why required_instance_extensions always lists two.
        for kind in [SurfaceKind::Wayland, SurfaceKind::Xcb, SurfaceKind::Xlib] {
            assert_ne!(kind.extension(), "VK_KHR_surface");
        }
    }
}
