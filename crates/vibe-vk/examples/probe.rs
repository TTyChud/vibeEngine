//! Prints the Vulkan devices on this machine and the sync tier each gets.
//!
//! Run with: cargo run -p vibe-vk --example probe

use vibe_vk::{QueueRequirements, enumerate_devices, load_entry, select_device};

fn main() {
    let entry = match unsafe { load_entry() } {
        Ok(e) => e,
        Err(e) => {
            eprintln!("vulkan loader: {e}");
            return;
        }
    };
    println!(
        "loader version: {}",
        vibe_vk::version_string(entry.loader_version)
    );

    let instance = match unsafe { vibe_vk::create_instance(&entry.ash, c"vibeEngine-probe", &[]) } {
        Ok(i) => i,
        Err(e) => {
            eprintln!("instance: {e}");
            return;
        }
    };

    let devices = match unsafe { enumerate_devices(&instance, entry.loader_version) } {
        Ok(d) => d,
        Err(e) => {
            eprintln!("enumerate: {e}");
            return;
        }
    };

    for d in &devices {
        println!(
            "\n{}\n  type      : {:?}\n  api       : {}\n  driver    : {}\n  memory    : {} MiB\n  sync tier : {}\n  bindless  : {:?}\n  extensions: {}",
            d.name,
            d.device_type,
            vibe_vk::version_string(d.api_version),
            d.driver_version,
            d.total_memory / (1024 * 1024),
            d.sync_tier,
            d.bindless,
            d.extensions.len()
        );
        for feature in [
            "VK_KHR_synchronization2",
            "VK_KHR_timeline_semaphore",
            "VK_KHR_dynamic_rendering",
            "VK_KHR_bindless_texture",
            "VK_EXT_descriptor_indexing",
            "VK_KHR_swapchain",
        ] {
            println!(
                "    {:32} {}",
                feature,
                d.extensions.iter().any(|e| e == feature)
            );
        }
    }

    match select_device(&devices, QueueRequirements::compute_only()) {
        Some(picked) => println!(
            "\nselected: {} ({}) tier {} barrier2={} timeline={}",
            picked.name,
            if picked.is_discrete {
                "discrete"
            } else {
                "integrated"
            },
            picked.sync_tier,
            picked.sync_tier.supports_barrier2(),
            picked.sync_tier.supports_timeline(),
        ),
        None => println!("\nno device selected"),
    }
}
