//! Window creation.

use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::{Window as WinitWindow, WindowId};

use crate::error::WindowError;

/// What to ask the windowing system for.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WindowDesc {
    /// Window title.
    pub title: String,
    /// Initial width in logical pixels.
    pub width: u32,
    /// Initial height in logical pixels.
    pub height: u32,
    /// Whether the window starts visible.
    pub visible: bool,
    /// Whether the window can be resized.
    pub resizable: bool,
}

impl Default for WindowDesc {
    fn default() -> Self {
        WindowDesc {
            title: "vibeEngine".to_string(),
            width: 1280,
            height: 720,
            visible: true,
            resizable: true,
        }
    }
}

/// What happened to the window since the last poll.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct WindowEvents {
    /// The window was closed.
    pub closed: bool,
    /// The window was resized or moved; the frame needs a new swapchain.
    pub resized: bool,
    /// The window's scale factor changed.
    pub scale_factor_changed: bool,
    /// True while the window has focus.
    pub focused: bool,
    /// The close button was pressed this poll.
    pub close_requested: bool,
}

impl WindowEvents {
    /// True when the frame loop should stop.
    pub fn should_exit(&self) -> bool {
        self.closed || self.close_requested
    }
}

/// The engine's window.
pub struct Window {
    /// The underlying winit window.
    pub inner: WinitWindow,
    events: WindowEvents,
}

impl Window {
    /// Wrap a winit window the engine did not create itself.
    pub fn from_inner(inner: WinitWindow) -> Window {
        Window {
            inner,
            events: WindowEvents::default(),
        }
    }

    /// The window handle, for Vulkan surface creation.
    pub fn window_handle(&self) -> Result<raw_window_handle::WindowHandle<'_>, WindowError> {
        self.inner
            .window_handle()
            .map_err(|_| WindowError::NoHandle("window"))
    }

    /// The display handle, for Vulkan surface creation.
    pub fn display_handle(&self) -> Result<raw_window_handle::DisplayHandle<'_>, WindowError> {
        self.inner
            .display_handle()
            .map_err(|_| WindowError::NoHandle("display"))
    }

    /// The window's size in physical pixels.
    pub fn size(&self) -> (u32, u32) {
        let s = self.inner.inner_size();
        (s.width, s.height)
    }

    /// The window's size in logical pixels, for UI layout.
    pub fn logical_size(&self) -> (f64, f64) {
        let s = self.inner.inner_size();
        (
            s.width as f64 / self.inner.scale_factor() as f64,
            s.height as f64 / self.inner.scale_factor() as f64,
        )
    }

    /// Take the events accumulated since the last poll.
    pub fn take_events(&mut self) -> WindowEvents {
        std::mem::take(&mut self.events)
    }

    /// Mark the window closed, for a caller driving events itself.
    pub fn close(&mut self) {
        self.events.closed = true;
    }

    /// Change the title.
    pub fn set_title(&self, title: &str) {
        self.inner.set_title(title);
    }
}

/// The winit application handler, which owns the window for the engine.
struct Engine<B> {
    window: Option<Window>,
    desc: WindowDesc,
    frame: Option<B>,
}

impl<B: FnMut(&mut Window) + 'static> ApplicationHandler for Engine<B> {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }
        let attrs = winit::window::WindowAttributes::default()
            .with_title(self.desc.title.clone())
            .with_inner_size(winit::dpi::LogicalSize::new(
                self.desc.width as f64,
                self.desc.height as f64,
            ))
            .with_visible(self.desc.visible)
            .with_resizable(self.desc.resizable);

        match event_loop.create_window(attrs) {
            Ok(inner) => {
                log::info!("window created: {}x{}", self.desc.width, self.desc.height);
                self.window = Some(Window {
                    inner,
                    events: WindowEvents::default(),
                });
            }
            Err(e) => log::error!("could not create a window: {e}"),
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let Some(window) = self.window.as_mut() else {
            return;
        };
        match event {
            WindowEvent::CloseRequested => {
                window.events.close_requested = true;
                event_loop.exit();
            }
            WindowEvent::Resized(_) | WindowEvent::ScaleFactorChanged { .. } => {
                // Either means the swapchain no longer matches the window.
                window.events.resized = true;
            }
            WindowEvent::Focused(focused) => window.events.focused = focused,
            WindowEvent::RedrawRequested => window.inner.request_redraw(),
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(window) = self.window.as_mut() {
            if let Some(frame) = self.frame.as_mut() {
                frame(window);
            }
        }
        if let Some(window) = self.window.as_ref() {
            window.inner.request_redraw();
        }
    }
}

/// Run a window until it closes, calling `frame` once per redraw.
///
/// This is the blocking entry point: the event loop owns the thread, so `frame`
/// runs from inside the loop and must not block for long. It returns only when
/// the window closes.
pub fn run<B>(desc: WindowDesc, frame: B) -> Result<(), WindowError>
where
    B: FnMut(&mut Window) + 'static,
{
    let event_loop = EventLoop::new().map_err(|e| WindowError::EventLoop(e.to_string()))?;
    event_loop.set_control_flow(ControlFlow::Poll);

    // The callback lives in the engine so the loop can reach it without a
    // borrow that would outlive `run_app`.
    let mut engine = Engine {
        window: None,
        desc,
        frame: Some(frame),
    };

    event_loop
        .run_app(&mut engine)
        .map_err(|e| WindowError::EventLoop(e.to_string()))?;
    Ok(())
}
