//! Proves the logical device's function table really resolves.
//!
//! ash 0.38 hands back an erased device whose entries are panicking stubs, so
//! the first command-buffer call segfaults. This creates a pool, allocates a
//! command buffer, records a `vkCmdDraw` and a dynamic-rendering scope, and
//! reports success. A fault here means the rebuild in `LogicalDevice::create`
//! regressed.
//!
//! Run with: WAYLAND_DISPLAY=wayland-1 cargo run -p vibe-vk --example devtable

use ash::vk;

fn main() {
    let entry = match unsafe { vibe_vk::load_entry() } {
        Ok(e) => e,
        Err(e) => {
            eprintln!("loader: {e}");
            return;
        }
    };
    let extensions: Vec<*const std::os::raw::c_char> = vec![
        c"VK_KHR_surface".as_ptr(),
        c"VK_KHR_wayland_surface".as_ptr(),
    ];
    let instance = match unsafe { vibe_vk::create_instance(&entry.ash, c"devtable", &extensions) } {
        Ok(i) => i,
        Err(e) => {
            eprintln!("instance: {e}");
            return;
        }
    };
    let devices = match unsafe { vibe_vk::enumerate_devices(&instance, entry.loader_version) } {
        Ok(d) => d,
        Err(e) => {
            eprintln!("devices: {e}");
            return;
        }
    };
    let Some(pd) = vibe_vk::select_device(&devices, vibe_vk::QueueRequirements::compute_only())
    else {
        eprintln!("no device");
        return;
    };
    let loader = ash::khr::surface::Instance::new(&entry.ash, &instance);
    let logical = match unsafe {
        vibe_vk::LogicalDevice::create(
            &instance,
            Some(&loader),
            pd.handle,
            None,
            vibe_vk::QueueRequirements::compute_only(),
        )
    } {
        Ok(d) => d,
        Err(e) => {
            eprintln!("device: {e}");
            return;
        }
    };

    let device = &logical.device;
    let family = logical
        .graphics_family()
        .or(logical.transfer_family())
        .unwrap_or(0);

    let pool_info = vk::CommandPoolCreateInfo {
        queue_family_index: family,
        flags: vk::CommandPoolCreateFlags::TRANSIENT,
        ..Default::default()
    };
    eprintln!("creating pool...");
    let pool = unsafe { device.create_command_pool(&pool_info, None) }.unwrap();
    eprintln!("pool ok");

    let alloc = vk::CommandBufferAllocateInfo {
        command_pool: pool,
        level: vk::CommandBufferLevel::PRIMARY,
        command_buffer_count: 1,
        ..Default::default()
    };
    let buffers = unsafe { device.allocate_command_buffers(&alloc) }.unwrap();
    eprintln!("buffers ok");

    let begin = vk::CommandBufferBeginInfo {
        flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
        ..Default::default()
    };
    eprintln!("beginning...");
    unsafe { device.begin_command_buffer(buffers[0], &begin) }.unwrap();
    eprintln!("begun");

    eprintln!("calling cmd_draw with 0 vertices...");
    unsafe { device.cmd_draw(buffers[0], 0, 1, 0, 0) };
    eprintln!("cmd_draw returned");

    // The frame path also begins dynamic rendering. Test that here.
    let rendering = ash::khr::dynamic_rendering::Device::new(&instance, device);
    let color = vk::RenderingAttachmentInfo {
        image_view: vk::ImageView::null(),
        image_layout: vk::ImageLayout::UNDEFINED,
        load_op: vk::AttachmentLoadOp::CLEAR,
        store_op: vk::AttachmentStoreOp::STORE,
        clear_value: vk::ClearValue {
            color: vk::ClearColorValue { float32: [0.0; 4] },
        },
        ..Default::default()
    };
    let info = vk::RenderingInfo {
        render_area: vk::Rect2D {
            offset: vk::Offset2D { x: 0, y: 0 },
            extent: vk::Extent2D {
                width: 64,
                height: 64,
            },
        },
        layer_count: 1,
        color_attachment_count: 1,
        p_color_attachments: &color,
        ..Default::default()
    };
    eprintln!("calling cmd_begin_rendering with a null view...");
    unsafe { rendering.cmd_begin_rendering(buffers[0], &info) };
    eprintln!("cmd_begin_rendering returned");
    unsafe { rendering.cmd_end_rendering(buffers[0]) };
    eprintln!("cmd_end_rendering returned");

    eprintln!("ending...");
    unsafe { device.end_command_buffer(buffers[0]) }.unwrap();
    eprintln!("ended -- the 1.0 command table resolves");
}
