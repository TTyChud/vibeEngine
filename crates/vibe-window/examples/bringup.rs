//! Brings up the whole stack against the real display and GPU, then reports
//! what it got. This is the check that a window, a Vulkan surface, a logical
//! device and a swapchain can all coexist on this machine.
//!
//! Run with: cargo run -p vibe-window --example bringup

use std::ffi::CStr;

use vibe_vk::context::{QueueRequirements, VkError};
use vibe_window::{WindowDesc, surface};

fn main() {
    // 1. Loader
    let entry = match unsafe { vibe_vk::load_entry() } {
        Ok(e) => e,
        Err(e) => {
            eprintln!("FAIL loader: {e}");
            return;
        }
    };
    println!(
        "[1] loader {}",
        vibe_vk::version_string(entry.loader_version)
    );

    // 2. Window on the real display
    let event_loop = match winit::event_loop::EventLoop::new() {
        Ok(e) => e,
        Err(e) => {
            eprintln!("FAIL event loop: {e}");
            return;
        }
    };

    // winit's event loop is consumed by run_app, so the window is handed back
    // through a shared slot rather than by borrowing a local.
    let out = std::sync::Arc::new(std::sync::Mutex::new(None));
    let sink = std::sync::Arc::clone(&out);
    let desc = WindowDesc::default();

    let _ = event_loop.run_app(&mut Probe { out: sink, desc });

    let taken = out.lock().ok().and_then(|mut g| g.take());
    let Some(window) = taken else {
        eprintln!("FAIL no window was created");
        return;
    };
    println!(
        "[2] window {}x{} (wayland)",
        window.size().0,
        window.size().1
    );

    // 3. Which surface extension does this display need?
    let raw = &window.inner;
    let kind = match surface::surface_kind(raw) {
        Ok(k) => k,
        Err(e) => {
            eprintln!("FAIL surface kind: {e}");
            return;
        }
    };
    println!("[3] platform {} needs {}", kind.name(), kind.extension());

    // 4. Instance with exactly those extensions
    let extensions = match surface::required_instance_extensions(raw) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("FAIL extensions: {e}");
            return;
        }
    };
    let instance =
        match unsafe { vibe_vk::create_instance(&entry.ash, c"vibeEngine-bringup", &extensions) } {
            Ok(i) => i,
            Err(e) => {
                eprintln!("FAIL instance: {e}");
                return;
            }
        };
    println!(
        "[4] instance created with {} extension(s)",
        extensions.len()
    );

    // 5. Surface
    let (surface, _) = match unsafe { surface::create_surface(&entry.ash, &instance, raw) } {
        Ok(s) => s,
        Err(e) => {
            eprintln!("FAIL surface: {e}");
            return;
        }
    };
    println!("[5] vulkan surface created");

    // 6. Device
    let devices = match unsafe { vibe_vk::enumerate_devices(&instance, entry.loader_version) } {
        Ok(d) => d,
        Err(e) => {
            eprintln!("FAIL devices: {e}");
            return;
        }
    };
    let Some(device) = vibe_vk::select_device(&devices, QueueRequirements::all()) else {
        eprintln!("FAIL no suitable device");
        return;
    };
    println!("[6] device {} tier {}", device.name, device.sync_tier);

    // 7. What the swapchain would actually be
    let surface_loader = ash::khr::surface::Instance::new(&entry.ash, &instance);
    let caps = match unsafe {
        surface_loader.get_physical_device_surface_capabilities(device.handle, surface)
    } {
        Ok(c) => c,
        Err(e) => {
            eprintln!("FAIL surface capabilities: {e}");
            return;
        }
    };
    let formats =
        match unsafe { surface_loader.get_physical_device_surface_formats(device.handle, surface) }
        {
            Ok(f) => f,
            Err(e) => {
                eprintln!("FAIL surface formats: {e}");
                return;
            }
        };
    let modes = match unsafe {
        surface_loader.get_physical_device_surface_present_modes(device.handle, surface)
    } {
        Ok(m) => m,
        Err(e) => {
            eprintln!("FAIL present modes: {e}");
            return;
        }
    };

    let picked_format = match vibe_vk::choose_surface_format(&formats) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("FAIL format choice: {e}");
            return;
        }
    };
    let picked_mode = vibe_vk::choose_present_mode(&modes);
    let images = vibe_vk::choose_image_count(&caps, 3);
    let (w, h) = window.size();

    println!("[7] swapchain would be:");
    println!("      extent        {}x{}", w, h);
    println!("      format        {:?}", picked_format.format);
    println!("      colour space  {:?}", picked_format.color_space);
    println!("      present mode  {picked_mode:?}");
    println!(
        "      images        {images} (min {} max {})",
        caps.min_image_count, caps.max_image_count
    );
    println!("      formats avail {}", formats.len());
    println!("      modes avail   {}", modes.len());

    // 8. Command pool, which needs a real logical device
    match unsafe {
        vibe_vk::LogicalDevice::create(
            &instance,
            Some(&surface_loader),
            device.handle,
            Some(surface),
            QueueRequirements::all(),
        )
    } {
        Ok(logical) => {
            println!(
                "[8] logical device created, {} extension(s), graphics queue {:?}",
                logical.features.extensions.len(),
                logical.graphics_queue()
            );
            let family = logical.graphics_family().unwrap_or(0);
            match unsafe { vibe_frame::CommandPool::new(&logical.device, family, 2) } {
                Ok(pool) => println!("[9] command pool with {} buffer(s)", pool.len()),
                Err(e) => eprintln!("FAIL command pool: {e}"),
            }
        }
        Err(e) => {
            eprintln!("FAIL logical device: {e}");
            // Re-derive the extension list to show what was asked for against
            // what the device reported, which is where the answer lives.
            match vibe_vk::device_extension_names(&instance, device.handle) {
                Ok(available) => {
                    let refs: Vec<&str> = available.iter().map(|s| s.as_str()).collect();
                    let chosen = vibe_vk::choose_extensions(device.sync_tier, &refs);
                    println!("    wanted:  {:?}", chosen.extensions);
                    println!("    missing: {:?}", chosen.missing);
                    for w in &chosen.extensions {
                        if !available.iter().any(|a| a == w) {
                            println!("    NOT ON DEVICE: {w}");
                        }
                    }
                }
                Err(e) => eprintln!("    could not list device extensions: {e}"),
            }
        }
    }

    let _ = VkError::NoPhysicalDevice;
    let _ = CStr::from_bytes_with_nul(b"x\0");
    let _ = out;
}

/// Opens a window, captures it, then exits.
struct Probe {
    out: std::sync::Arc<std::sync::Mutex<Option<vibe_window::Window>>>,
    desc: WindowDesc,
}

impl winit::application::ApplicationHandler for Probe {
    fn window_event(
        &mut self,
        _event_loop: &winit::event_loop::ActiveEventLoop,
        _id: winit::window::WindowId,
        _event: winit::event::WindowEvent,
    ) {
    }

    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        if self.out.lock().map(|g| g.is_some()).unwrap_or(true) {
            return;
        }
        let attrs = winit::window::WindowAttributes::default()
            .with_title(self.desc.title.clone())
            .with_inner_size(winit::dpi::LogicalSize::new(
                self.desc.width as f64,
                self.desc.height as f64,
            ))
            .with_visible(self.desc.visible);
        if let Ok(w) = event_loop.create_window(attrs) {
            if let Ok(mut slot) = self.out.lock() {
                *slot = Some(vibe_window::Window::from_inner(w));
            }
            event_loop.exit();
        }
    }
}
