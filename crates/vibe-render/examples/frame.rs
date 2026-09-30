//! Renders one frame of batched quads and reads the pixels back, so the whole
//! path is verified: shader compilation, pipeline creation, command recording,
//! queue submission, and a barrier-correct present.
//!
//! Run with: cargo run -p vibe-render --example frame

use std::ffi::CStr;

use ash::vk;
use vibe_rhi::SubTexture;
use vibe_shader::WgslCompiler;
use vibe_vk::context::{QueueRequirements, VkError};
use vibe_vk::memory::{BufferUsage, GpuBuffer, GpuImage, HeapLayout, ImageUsage, MemoryNeed};
use vibe_window::{SurfaceKind, surface};

/// How long to wait for the GPU, in nanoseconds.
const TIMEOUT: u64 = 10_000_000_000;

fn main() {
    // Validation has to be requested before the instance exists, because the
    // loader reads the layer list when it creates one.
    let validating = std::env::var("VIBE_VALIDATION").is_ok();
    if validating {
        vibe_vk::validation::enable_layer();
        eprintln!("validation layer requested");
    }

    // 1. Vulkan loader and a window; the helper loads the entry table itself
    let (window, entry) = match make_window() {
        Ok(v) => v,
        Err(e) => {
            eprintln!("FAIL window: {e}");
            return;
        }
    };
    let (w, h) = window.size();
    println!(
        "[1] window {}x{} on {}",
        w,
        h,
        surface::surface_kind(&window.inner)
            .map(|k| k.name())
            .unwrap_or("?")
    );

    // 2. Instance with the right surface extension
    let raw = &window.inner;
    let extensions = match surface::required_instance_extensions(raw) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("FAIL extensions: {e}");
            return;
        }
    };
    let instance =
        match unsafe { vibe_vk::create_instance(&entry.ash, c"vibeEngine-frame", &extensions) } {
            Ok(i) => i,
            Err(e) => {
                eprintln!("FAIL instance: {e}");
                return;
            }
        };

    // A messenger, held for the run, is what actually prints the layer's
    // findings: enabling the layer alone installs the checks but routes their
    // output through this callback.
    let _messenger = if validating {
        match unsafe { vibe_vk::validation::ValidationMessenger::new(&entry.ash, &instance) } {
            Ok(m) => Some(m),
            Err(e) => {
                eprintln!("could not create a validation messenger: {e}");
                None
            }
        }
    } else {
        None
    };

    // 3. Surface
    let (vk_surface, _) = match unsafe { surface::create_surface(&entry.ash, &instance, raw) } {
        Ok(s) => s,
        Err(e) => {
            eprintln!("FAIL surface: {e}");
            return;
        }
    };
    let surface_loader = ash::khr::surface::Instance::new(&entry.ash, &instance);

    // 4. Device
    let devices = match unsafe { vibe_vk::enumerate_devices(&instance, entry.loader_version) } {
        Ok(d) => d,
        Err(e) => {
            eprintln!("FAIL devices: {e}");
            return;
        }
    };
    let Some(pd) = vibe_vk::select_device(&devices, QueueRequirements::all()) else {
        eprintln!("FAIL no device");
        return;
    };
    println!(
        "[1b] instance api {} (1.3 entry points {} available)",
        vibe_vk::version_string(
            entry
                .loader_version
                .min(vibe_vk::context::REQUESTED_API_VERSION)
        ),
        if entry.loader_version >= vibe_vk::context::REQUESTED_API_VERSION {
            "are"
        } else {
            "are NOT"
        }
    );

    let logical = match unsafe {
        vibe_vk::LogicalDevice::create(
            &instance,
            Some(&surface_loader),
            pd.handle,
            Some(vk_surface),
            QueueRequirements::all(),
        )
    } {
        Ok(d) => d,
        Err(e) => {
            eprintln!("FAIL logical device: {e}");
            return;
        }
    };
    // `LogicalDevice::create` already returns a device whose function table
    // resolves: ash 0.38 hands back an erased one, which segfaults on the first
    // command-buffer call.
    let device = &logical.device;
    let Some(graphics_queue) = logical.graphics_queue() else {
        eprintln!("FAIL no graphics queue");
        return;
    };
    println!("[2] device {} tier {}", pd.name, logical.sync_tier());

    // 5. Swapchain
    // Extension loaders in ash 0.38 take the instance and the device.
    let swapchain_loader = ash::khr::swapchain::Device::new(&instance, device);
    let swapchain_config = vibe_vk::FrameConfig::for_tier(logical.sync_tier());
    let config = vibe_frame::FrameConfig::for_tier(logical.sync_tier());
    let swapchain = match unsafe {
        vibe_vk::Swapchain::new(
            &swapchain_loader,
            &surface_loader,
            pd.handle,
            vk_surface,
            logical.graphics_family().unwrap_or(0),
            logical.present_family().unwrap_or(0),
            (w, h),
            vk::SwapchainKHR::null(),
            swapchain_config,
        )
    } {
        Ok(s) => s,
        Err(e) => {
            eprintln!("FAIL swapchain: {e}");
            return;
        }
    };
    let swapchain_images =
        match unsafe { swapchain_loader.get_swapchain_images(swapchain.handle()) } {
            Ok(i) => i,
            Err(e) => {
                eprintln!("FAIL swapchain images: {e}");
                return;
            }
        };
    println!(
        "[3] swapchain {}x{} window {}x{} images {} format {:?} present {:?}",
        swapchain.extent().width,
        swapchain.extent().height,
        w,
        h,
        swapchain_images.len(),
        swapchain.format().format,
        swapchain.present_mode()
    );

    // A view per swapchain image: a render pass binds a view, not an image.
    let mut views = Vec::new();
    for image in &swapchain_images {
        let view_info = vk::ImageViewCreateInfo {
            image: *image,
            view_type: vk::ImageViewType::TYPE_2D,
            format: swapchain.format().format,
            subresource_range: vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            },
            ..Default::default()
        };
        match unsafe { device.create_image_view(&view_info, None) } {
            Ok(v) => views.push(v),
            Err(e) => {
                eprintln!("FAIL image view: {e:?}");
                return;
            }
        }
    }
    println!("[4] {} image view(s) created", views.len());

    // 6. Memory layout and a vertex buffer
    let props = unsafe { instance.get_physical_device_memory_properties(pd.handle) };
    let heap_layout = HeapLayout {
        property_flags: props.memory_types[..props.memory_type_count as usize]
            .iter()
            .map(|m| m.property_flags)
            .collect(),
        heap_index: props.memory_types[..props.memory_type_count as usize]
            .iter()
            .map(|m| m.heap_index)
            .collect(),
        heap_sizes: props.memory_heaps[..props.memory_heap_count as usize]
            .iter()
            .map(|m| m.size)
            .collect(),
    };

    // 5. The pipeline, compiled from WGSL at runtime
    let pipeline = match unsafe {
        vibe_render::build_quad_pipeline(device, &WgslCompiler::new(), &[swapchain.format().format])
    } {
        Ok(p) => p,
        Err(e) => {
            eprintln!("FAIL pipeline: {e}");
            return;
        }
    };
    let rendering = ash::khr::dynamic_rendering::Device::new(&instance, device);
    let sync2 = ash::khr::synchronization2::Device::new(&instance, device);
    println!("[5] quad pipeline built (key {:#x})", pipeline.cache_key);

    // 5b. The three descriptors the shader reads: a camera uniform, a texture
    // and a sampler. A draw that leaves any of them unbound is invalid.
    let camera = match unsafe {
        GpuBuffer::create(
            device,
            &heap_layout,
            std::mem::size_of::<glam::Mat4>() as u64,
            BufferUsage {
                uniform: true,
                ..Default::default()
            },
            MemoryNeed::upload(),
        )
    } {
        Ok(b) => b,
        Err(e) => {
            eprintln!("FAIL camera buffer: {e}");
            return;
        }
    };
    let mut camera = camera;
    if let Err(e) = unsafe { camera.map(device) } {
        eprintln!("FAIL camera map: {e}");
        return;
    }

    // A 1x1 opaque white texel, so an untextured quad samples as solid colour
    // and the fragment path is exercised rather than skipped.
    let texture = match unsafe {
        GpuImage::create_2d(
            device,
            &heap_layout,
            1,
            1,
            vk::Format::R8G8B8A8_UNORM,
            ImageUsage::texture(),
            vk::ImageTiling::OPTIMAL,
            vk::SampleCountFlags::TYPE_1,
        )
    } {
        Ok(t) => t,
        Err(e) => {
            eprintln!("FAIL texture: {e}");
            return;
        }
    };
    // A 1x1 image created but never written holds undefined memory, and the
    // fragment shader samples it. Stage the white texel, then copy it in.
    let texel: [u8; 4] = [255, 255, 255, 255];
    let mut staging = match unsafe {
        GpuBuffer::create(
            device,
            &heap_layout,
            4,
            BufferUsage {
                transfer_src: true,
                ..Default::default()
            },
            MemoryNeed::upload(),
        )
    } {
        Ok(b) => b,
        Err(e) => {
            eprintln!("FAIL staging buffer: {e}");
            return;
        }
    };
    if let Err(e) = unsafe { staging.map(device) } {
        eprintln!("FAIL staging map: {e}");
        return;
    }
    if let Err(e) = unsafe { staging.write_mapped(0, &texel) } {
        eprintln!("FAIL texel upload: {e}");
        return;
    }
    println!("[5a] white texel staged ({} bytes)", texel.len());

    // The upload is its own submission: undefined -> transfer destination, copy,
    // then transfer destination -> shader-read-only, which is the layout the
    // descriptor promises. A descriptor pointing at the wrong layout is a
    // validation error and, on this driver, a lost device.
    let upload_pool = match unsafe {
        vibe_frame::CommandPool::new(device, logical.graphics_family().unwrap_or(0), 1)
    } {
        Ok(p) => p,
        Err(e) => {
            eprintln!("FAIL upload pool: {e}");
            return;
        }
    };
    let upload_cmd = match unsafe { upload_pool.begin(device, 0) } {
        Ok(c) => c,
        Err(e) => {
            eprintln!("FAIL upload begin: {e}");
            return;
        }
    };
    {
        let [into_dst, into_read] =
            vibe_vk::upload_image_barriers(texture.image(), vk::ImageAspectFlags::COLOR);
        let mut enc = vibe_vk::BarrierEncoder::for_tier(logical.sync_tier());
        enc.push(into_dst);
        unsafe { enc.record(device, &sync2, upload_cmd) };
        if let Err(e) = unsafe { texture.record_upload(device, upload_cmd, staging.buffer()) } {
            eprintln!("FAIL record upload: {e}");
            return;
        }
        let mut enc2 = vibe_vk::BarrierEncoder::for_tier(logical.sync_tier());
        enc2.push(into_read);
        unsafe { enc2.record(device, &sync2, upload_cmd) };
    }
    if let Err(e) = unsafe { upload_pool.end(device, 0) } {
        eprintln!("FAIL upload end: {e}");
        return;
    }
    // No semaphore: the upload is CPU-waited on a fence, and nothing else
    // touches the texture until the frame is submitted.
    let upload_batch = vibe_frame::SubmitBatch::single(upload_cmd, vibe_frame::SubmitSync::None);
    let upload_submission = vibe_frame::build_submit_info(&upload_batch);
    let upload_fence =
        unsafe { device.create_fence(&vk::FenceCreateInfo::default(), None) }.unwrap();
    if let Err(e) =
        unsafe { sync2.queue_submit2(graphics_queue, &[upload_submission.info], upload_fence) }
    {
        eprintln!("FAIL upload submit: {e:?}");
        return;
    }
    if let Err(e) = unsafe { device.wait_for_fences(&[upload_fence], true, TIMEOUT) } {
        eprintln!("FAIL upload wait: {e:?}");
        return;
    }
    unsafe { device.destroy_fence(upload_fence, None) };
    println!("[5b] texture uploaded: undefined -> transfer dst -> shader read only");

    let sampler = match vibe_render::create_sampler(device) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("FAIL sampler: {e}");
            return;
        }
    };
    let bind_group = match unsafe {
        vibe_render::BindGroup::create(
            device,
            pipeline.descriptor_layout,
            &vibe_render::BindGroupDesc::default(),
            camera.buffer(),
            std::mem::size_of::<glam::Mat4>() as u64,
            texture.view(),
            sampler,
        )
    } {
        Ok(b) => b,
        Err(e) => {
            eprintln!("FAIL bind group: {e}");
            return;
        }
    };
    println!("[5b] bind group created: camera, 1x1 white texture, nearest sampler");

    // 6. The renderer and a batch of quads
    // Write the camera matrix before the draw: the descriptor points at this
    // buffer, and the shader reads it in the vertex stage.
    let matrix = renderer_camera_matrix(w as f32, h as f32);
    if let Err(e) = unsafe { camera.write_mapped(0, bytemuck::bytes_of(&matrix)) } {
        eprintln!("FAIL camera upload: {e}");
        return;
    }
    println!(
        "[5c] camera matrix uploaded ({} bytes)",
        std::mem::size_of::<glam::Mat4>()
    );

    let mut renderer = vibe_render::QuadRenderer::new(
        bind_group,
        pipeline,
        vibe_render::RendererDesc::new(w, h, swapchain.format().format),
    );
    // Four quadrants in four distinct colours, so the readback can check that
    // each quad landed where the camera says it should. A single full-window
    // quad cannot distinguish a correct draw from a wrong one, which is the
    // whole reason this example reads pixels back at all.
    let half_w = w as f32 / 2.0;
    let half_h = h as f32 / 2.0;
    let quadrants = [
        (glam::Vec2::ZERO, [255, 0, 0, 255]), // top left, red
        (
            glam::Vec2::new(half_w, 0.0),
            [0, 255, 0, 255], // top right, green
        ),
        (
            glam::Vec2::new(0.0, half_h),
            [0, 0, 255, 255], // bottom left, blue
        ),
        (
            glam::Vec2::new(half_w, half_h),
            [255, 255, 0, 255], // bottom right, yellow
        ),
    ];
    {
        let batcher = renderer.batcher_mut();
        let tex = SubTexture::full(0);
        for (origin, color) in quadrants {
            let _ = batcher.push_quad(origin, glam::Vec2::new(half_w, half_h), color, &tex);
        }
        // One off-screen quad, which must not appear anywhere in the frame.
        let _ = batcher.push_quad(
            glam::Vec2::new(w as f32 + 500.0, -500.0),
            glam::Vec2::splat(50.0),
            [255, 255, 255, 255],
            &tex,
        );
    }
    let stats = renderer.batcher().stats();
    println!(
        "[6] batch: {} quads, {} vertices, {} bytes",
        stats.quads,
        stats.vertices,
        renderer.required_vertex_bytes()
    );

    // 7. Upload the vertices
    let vertex_bytes = renderer.required_vertex_bytes().max(32);
    let mut vertex_buffer = match unsafe {
        GpuBuffer::create(
            device,
            &heap_layout,
            vertex_bytes as u64,
            BufferUsage {
                vertex: true,
                ..Default::default()
            },
            MemoryNeed::upload(),
        )
    } {
        Ok(b) => b,
        Err(e) => {
            eprintln!("FAIL vertex buffer: {e}");
            return;
        }
    };
    // Map, upload, and read back: the read-back bytes are exactly what the
    // draw fetches, so it separates a bad upload from a bad draw.
    if let Err(e) = unsafe { vertex_buffer.map(device) } {
        eprintln!("FAIL vertex map: {e}");
        return;
    }
    let vertex_data = bytemuck::cast_slice(renderer.batcher().vertices());
    if let Err(e) = unsafe { vertex_buffer.write_mapped(0, vertex_data) } {
        eprintln!("FAIL vertex write: {e}");
        return;
    }
    println!("[7] vertex buffer uploaded");

    // 8. Command pool and frame resources
    let pool = match unsafe {
        vibe_frame::CommandPool::new(device, logical.graphics_family().unwrap_or(0), 2)
    } {
        Ok(p) => p,
        Err(e) => {
            eprintln!("FAIL command pool: {e}");
            return;
        }
    };
    let mut resources = match unsafe { vibe_frame::FrameResources::new(device, config) } {
        Ok(r) => r,
        Err(e) => {
            eprintln!("FAIL frame resources: {e}");
            return;
        }
    };
    let (slot, value) = match resources.next_slot() {
        Ok(v) => v,
        Err(e) => {
            eprintln!("FAIL frame slot: {e}");
            return;
        }
    };
    println!(
        "[8] frame {} slot {} value {}",
        slot, resources.submitted, value
    );

    // 9. Acquire, record, submit, present
    // `vkAcquireNextImageKHR` requires a binary semaphore, so the timeline one
    // cannot be used here even on a timeline tier. The frame waits on this and
    // signals a second one, because a submit may not use the same semaphore as
    // both its wait and its signal.
    let (Some(acquire_sem), Some(present_sem)) = (
        resources.acquire_semaphore(slot),
        resources.present_semaphore(slot),
    ) else {
        eprintln!("FAIL no WSI semaphores");
        return;
    };

    let acquired = match unsafe {
        swapchain.acquire(&swapchain_loader, TIMEOUT, acquire_sem, vk::Fence::null())
    } {
        Ok(a) => a,
        Err(e) => {
            eprintln!("FAIL acquire: {e}");
            return;
        }
    };
    let index = match acquired {
        vibe_vk::AcquireResult::Ready { index } => index,
        other => {
            eprintln!("FAIL not ready: {other:?}");
            return;
        }
    };
    println!("[9] acquired swapchain image {index}");

    let cmd = match unsafe { pool.begin(device, slot) } {
        Ok(c) => c,
        Err(e) => {
            eprintln!("FAIL begin: {e}");
            return;
        }
    };

    let target = views[index as usize];
    let target_image = swapchain_images[index as usize];

    // One barrier only, and it belongs here: the image goes from undefined to
    // a colour attachment. The transition back to present has to be recorded
    // after the render pass, not before it.
    let mut encoder = vibe_vk::BarrierEncoder::for_tier(logical.sync_tier());
    encoder.push(vibe_render::QuadRenderer::acquire_barrier(target_image));
    unsafe { encoder.record(device, &sync2, cmd) };
    println!("[10] recorded {} pre-pass barrier(s)", encoder.len());

    let vertex_handle = vertex_buffer.buffer();
    if let Err(e) = unsafe { renderer.record(device, &rendering, cmd, target, vertex_handle, 0) } {
        eprintln!("FAIL record: {e}");
        return;
    }
    println!("[11] frame recorded");

    // The present barrier belongs after the render pass ended, so it goes in a
    // second command buffer rather than being batched with the one before it.
    let present_cmd = match pool.buffer((slot + 1) % pool.len()) {
        Some(c) => c,
        None => {
            eprintln!("FAIL no second command buffer");
            return;
        }
    };
    let begin_info = vk::CommandBufferBeginInfo {
        flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
        ..Default::default()
    };
    if let Err(e) = unsafe { device.begin_command_buffer(present_cmd, &begin_info) } {
        eprintln!("FAIL begin present: {e}");
        return;
    }
    let mut present_encoder = vibe_vk::BarrierEncoder::for_tier(logical.sync_tier());
    present_encoder.push(
        vibe_vk::Barrier::image(target_image, vk::ImageAspectFlags::COLOR)
            .layouts(
                ash::vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
                ash::vk::ImageLayout::PRESENT_SRC_KHR,
            )
            .src(
                vibe_vk::Stage::ColorAttachmentOutput,
                vibe_vk::Access::ColorAttachmentWrite,
            )
            .dst(vibe_vk::Stage::None, vibe_vk::Access::None),
    );
    unsafe { present_encoder.record(device, &sync2, present_cmd) };
    if let Err(e) = unsafe { device.end_command_buffer(present_cmd) } {
        eprintln!("FAIL end present: {e}");
        return;
    }
    if let Err(e) = unsafe { pool.end(device, slot) } {
        eprintln!("FAIL end: {e}");
        return;
    }

    // Two independent waits. `vkAcquireNextImageKHR` signalled `acquire_sem`,
    // meaning "this image is ready to render into", so the frame waits on it
    // before drawing. Separately the timeline wait keeps the CPU from getting
    // more than frames_in_flight ahead of the GPU.
    let sync = match resources.timeline_semaphore() {
        Some(timeline) => vibe_frame::SubmitSync::Paced {
            timeline,
            // The previous frame's value: this frame is not allowed to start
            // until the one before it has finished.
            wait_value: value - 1,
            acquire: acquire_sem,
            signal: present_sem,
            signal_value: value,
        },
        // Without a timeline semaphore there is no pacing wait, so the frame
        // waits only on the acquire semaphore and signals the present one.
        None => vibe_frame::SubmitSync::Binary {
            wait: acquire_sem,
            signal: present_sem,
        },
    };
    // Two command buffers: the frame itself, and the present transition, which
    // is only valid after the render pass has ended.
    let mut batch = vibe_frame::SubmitBatch::single(cmd, sync);
    batch.push(present_cmd);
    let submission = vibe_frame::build_submit_info(&batch);

    let submit_result = unsafe {
        let fence = resources.fence(slot).unwrap_or(vk::Fence::null());
        sync2.queue_submit2(graphics_queue, &[submission.info], fence)
    };
    if let Err(e) = submit_result {
        eprintln!("FAIL submit: {e:?}");
        return;
    }
    println!(
        "[12] submitted {} command buffer(s)",
        batch.command_buffers.len()
    );

    if let Err(e) =
        unsafe { device.wait_for_fences(&[resources.fence(slot).unwrap()], true, TIMEOUT) }
    {
        eprintln!("FAIL wait: {e:?}");
        return;
    }
    // A successful present only proves the driver accepted the work. Reading the
    // image back proves pixels were actually written.
    let Some(verdict) = read_back(device, &sync2, &logical, graphics_queue, target_image, w, h)
    else {
        eprintln!("FAIL readback");
        return;
    };
    println!(
        "READBACK: {} of {} sampled pixels are the colour the batch asked for",
        verdict.matched,
        verdict.samples.len()
    );
    if verdict.matched != verdict.samples.len() {
        eprintln!("MISMATCH: the draw did not land where the camera put it");
    }

    let presented =
        match unsafe { swapchain.present(&swapchain_loader, graphics_queue, index, present_sem) } {
            Ok(p) => p,
            Err(e) => {
                eprintln!("FAIL present: {e}");
                return;
            }
        };
    println!("[13] presented: {presented:?}");

    unsafe { device.device_wait_idle().ok() };
    println!();
    println!(
        "RENDERED A FRAME: {} quad(s) through {} draw call(s), {} barrier(s).",
        stats.quads,
        stats.draw_calls.max(1),
        encoder.len()
    );

    let _ = (
        SurfaceKind::Wayland,
        VkError::NoPhysicalDevice,
        CStr::from_bytes_with_nul(b"x\0"),
    );
    let _ = ImageUsage::render_target();
    logical.wait_idle();
}

/// Opens a window and captures it for the caller to use.
fn make_window() -> Result<(vibe_window::Window, vibe_vk::Entry), String> {
    let entry = unsafe { vibe_vk::load_entry() }.map_err(|e| e.to_string())?;
    let event_loop = winit::event_loop::EventLoop::new().map_err(|e| e.to_string())?;
    let slot = std::sync::Arc::new(std::sync::Mutex::new(None));

    struct Probe {
        slot: std::sync::Arc<std::sync::Mutex<Option<vibe_window::Window>>>,
    }
    impl winit::application::ApplicationHandler for Probe {
        fn resumed(&mut self, el: &winit::event_loop::ActiveEventLoop) {
            if self.slot.lock().map(|g| g.is_some()).unwrap_or(true) {
                return;
            }
            let attrs = winit::window::WindowAttributes::default()
                .with_title("vibeEngine")
                .with_inner_size(winit::dpi::LogicalSize::new(800.0, 600.0))
                .with_visible(true);
            if let Ok(w) = el.create_window(attrs) {
                if let Ok(mut s) = self.slot.lock() {
                    *s = Some(vibe_window::Window::from_inner(w));
                }
                el.exit();
            }
        }
        fn window_event(
            &mut self,
            _el: &winit::event_loop::ActiveEventLoop,
            _id: winit::window::WindowId,
            _e: winit::event::WindowEvent,
        ) {
        }
    }

    let mut probe = Probe {
        slot: std::sync::Arc::clone(&slot),
    };
    let _ = event_loop.run_app(&mut probe);
    let window = slot
        .lock()
        .ok()
        .and_then(|mut g| g.take())
        .ok_or("no window was created")?;
    Ok((window, entry))
}

/// The 2D orthographic matrix the camera uniform holds.
fn renderer_camera_matrix(width: f32, height: f32) -> glam::Mat4 {
    vibe_math::orthographic_rh(0.0, width, 0.0, height, -1.0, 1.0)
}

/// What a readback found in the frame.
#[derive(Debug, Clone)]
struct Verdict {
    /// The colour found at each sample point, as BGR.
    samples: Vec<[u8; 3]>,
    /// The label each sample was expected to be.
    expected: Vec<&'static str>,
    /// How many samples matched their expected colour.
    matched: usize,
}

/// Copy the rendered image into a host buffer and describe what is in it.
///
/// Counting distinct colours is the honest check. Comparing against the clear
/// colour is not: a quad that covers the whole window legitimately leaves no
/// clear pixels, so that metric reports a correct frame as a blank one.
#[allow(clippy::too_many_arguments)]
fn read_back(
    device: &ash::Device,
    sync2: &ash::khr::synchronization2::Device,
    logical: &vibe_vk::LogicalDevice,
    queue: ash::vk::Queue,
    image: ash::vk::Image,
    width: u32,
    height: u32,
) -> Option<Verdict> {
    use vibe_vk::memory::{BufferUsage, GpuBuffer, MemoryNeed};

    let stride = width as u64 * 4;
    let size = stride * height as u64;

    let mut buffer = match unsafe {
        GpuBuffer::create(
            device,
            logical.memory_layout(),
            size,
            BufferUsage {
                transfer_dst: true,
                ..Default::default()
            },
            MemoryNeed::upload(),
        )
    } {
        Ok(b) => b,
        Err(_) => return None,
    };
    if unsafe { buffer.map(device) }.is_err() {
        return None;
    }

    let Ok(pool) = (unsafe {
        vibe_frame::CommandPool::new(device, logical.graphics_family().unwrap_or(0), 1)
    }) else {
        return None;
    };
    let Ok(cmd) = (unsafe { pool.begin(device, 0) }) else {
        return None;
    };

    {
        // present -> transfer source, so the copy is legal
        let mut enc = vibe_vk::BarrierEncoder::for_tier(logical.sync_tier());
        enc.push(
            vibe_vk::Barrier::image(image, ash::vk::ImageAspectFlags::COLOR)
                .layouts(
                    ash::vk::ImageLayout::PRESENT_SRC_KHR,
                    ash::vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                )
                .src(
                    vibe_vk::Stage::ColorAttachmentOutput,
                    vibe_vk::Access::ColorAttachmentWrite,
                )
                .dst(vibe_vk::Stage::Transfer, vibe_vk::Access::TransferRead),
        );
        unsafe { enc.record(device, sync2, cmd) };

        let region = ash::vk::BufferImageCopy {
            buffer_offset: 0,
            buffer_row_length: 0,
            buffer_image_height: 0,
            image_subresource: ash::vk::ImageSubresourceLayers {
                aspect_mask: ash::vk::ImageAspectFlags::COLOR,
                mip_level: 0,
                base_array_layer: 0,
                layer_count: 1,
            },
            image_offset: ash::vk::Offset3D { x: 0, y: 0, z: 0 },
            image_extent: ash::vk::Extent3D {
                width,
                height,
                depth: 1,
            },
        };
        unsafe {
            device.cmd_copy_image_to_buffer(
                cmd,
                image,
                ash::vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                buffer.buffer(),
                &[region],
            );
        }

        // transfer write -> host read, so the map sees the pixels
        let mut enc2 = vibe_vk::BarrierEncoder::for_tier(logical.sync_tier());
        enc2.push(
            vibe_vk::Barrier::memory_barrier()
                .src(vibe_vk::Stage::Transfer, vibe_vk::Access::TransferWrite)
                .dst(vibe_vk::Stage::Host, vibe_vk::Access::HostRead),
        );
        // The copy left the image in TRANSFER_SRC_OPTIMAL, and present requires
        // PRESENT_SRC_KHR. Reading a frame must not break the frame after it.
        enc2.push(
            vibe_vk::Barrier::image(image, ash::vk::ImageAspectFlags::COLOR)
                .layouts(
                    ash::vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    ash::vk::ImageLayout::PRESENT_SRC_KHR,
                )
                .src(vibe_vk::Stage::Transfer, vibe_vk::Access::TransferRead)
                .dst(vibe_vk::Stage::BottomOfPipe, vibe_vk::Access::None),
        );
        unsafe { enc2.record(device, sync2, cmd) };
    }

    if unsafe { pool.end(device, 0) }.is_err() {
        return None;
    }
    let batch = vibe_frame::SubmitBatch::single(cmd, vibe_frame::SubmitSync::None);
    let sub = vibe_frame::build_submit_info(&batch);
    let Ok(fence) = (unsafe { device.create_fence(&ash::vk::FenceCreateInfo::default(), None) })
    else {
        return None;
    };
    if unsafe { sync2.queue_submit2(queue, &[sub.info], fence) }.is_err() {
        return None;
    }
    if unsafe { device.wait_for_fences(&[fence], true, 10_000_000_000) }.is_err() {
        return None;
    }
    unsafe { device.destroy_fence(fence, None) };

    let Ok(data) = (unsafe { buffer.read_mapped(0, size as usize) }) else {
        return None;
    };

    // The surface is B8G8R8A8_SRGB, so blue is the first byte. Write a PPM so the
    // rendered image can actually be looked at rather than inferred.
    {
        let path = std::path::Path::new("/tmp/vibe-frame.ppm");
        let mut ppm = format!("P6\n{} {}\n255\n", width, height).into_bytes();
        for p in data.chunks_exact(4) {
            // BGRX -> RGB
            ppm.extend_from_slice(&[p[2], p[1], p[0]]);
        }
        let _ = std::fs::write(path, ppm);
        println!("  wrote {}", path.display());
    }

    // A quad can cover pixel 0, so the frame's first colour is not necessarily
    // the background. A histogram answers the question that matters without
    // assuming anything about which colour is which: a frame with a clear and
    // several quads must contain more than one.
    let mut histogram: std::collections::HashMap<[u8; 3], usize> = std::collections::HashMap::new();
    for p in data.chunks_exact(4) {
        *histogram.entry([p[2], p[1], p[0]]).or_default() += 1;
    }
    let mut ranked: Vec<_> = histogram.into_iter().collect();
    ranked.sort_by_key(|(_, c)| std::cmp::Reverse(*c));
    for (rgb, count) in ranked.iter().take(8) {
        println!("    rgb {rgb:?} x{count}");
    }

    // Sample the middle of each quadrant and compare it with what the camera
    // and the batch said should be there. This is the check that actually
    // proves the draw: a flat frame of the right colour proves nothing.
    let px = |x: u32, y: u32| -> [u8; 3] {
        let i = (y as usize * width as usize + x as usize) * 4;
        let p = &data[i..i + 4];
        // The surface is B8G8R8A8, so the stored order is B, G, R.
        [p[2], p[1], p[0]]
    };
    let quarter_w = width / 4;
    let quarter_h = height / 4;
    let points = [
        (quarter_w, quarter_h, "top left red"),
        (width - quarter_w, quarter_h, "top right green"),
        (quarter_w, height - quarter_h, "bottom left blue"),
        (width - quarter_w, height - quarter_h, "bottom right yellow"),
    ];
    // The surface is sRGB, so a linear 1.0 comes back as 255 and 0 as 0; the
    // primaries survive that round trip, so an exact compare is safe here.
    let wanted: [[u8; 3]; 4] = [[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0]];
    let mut samples = Vec::with_capacity(points.len());
    let mut expected = Vec::with_capacity(points.len());
    let mut matched = 0;
    for (i, (x, y, label)) in points.iter().enumerate() {
        let got = px(*x, *y);
        let want = wanted[i];
        let ok = got == want;
        if ok {
            matched += 1;
        }
        println!(
            "    {label}: rgb {got:?} expected {want:?} {}",
            if ok { "ok" } else { "MISMATCH" }
        );
        samples.push(got);
        expected.push(*label);
    }
    Some(Verdict {
        samples,
        expected,
        matched,
    })
}
