//! Probes what this machine's Vulkan setup offers: device, tier, and which
//! window-system surface extensions the loader accepted.
//!
//! Run with: RUST_LOG=info cargo run -p vibe-vk --example surface-probe

use vibe_vk::{QueueRequirements, create_instance, enumerate_devices, load_entry, select_device};

fn main() {
    let entry = match unsafe { load_entry() } {
        Ok(e) => e,
        Err(e) => {
            eprintln!("loader: {e}");
            return;
        }
    };
    println!("loader {}", vibe_vk::version_string(entry.loader_version));

    let surface_extensions: Vec<*const std::os::raw::c_char> = vec![
        c"VK_KHR_surface".as_ptr(),
        c"VK_KHR_wayland_surface".as_ptr(),
        c"VK_KHR_xcb_surface".as_ptr(),
        c"VK_KHR_xlib_surface".as_ptr(),
    ];

    // create_instance logs which extensions it actually enabled; the debug
    // output above this line is the report.
    let instance =
        match unsafe { create_instance(&entry.ash, c"vibeEngine-probe", &surface_extensions) } {
            Ok(i) => i,
            Err(e) => {
                eprintln!("instance: {e}");
                return;
            }
        };

    let devices = match unsafe { enumerate_devices(&instance, entry.loader_version) } {
        Ok(d) => d,
        Err(e) => {
            eprintln!("devices: {e}");
            return;
        }
    };

    for d in &devices {
        println!(
            "{}: tier {} bindless {:?} extensions {}",
            d.name,
            d.sync_tier,
            d.bindless,
            d.extensions.len()
        );
    }

    match select_device(&devices, QueueRequirements::all()) {
        Some(p) => println!("selected {}", p.name),
        None => println!("no suitable device"),
    }

    let _ = ash::khr::surface::Instance::new(&entry.ash, &instance);
}
