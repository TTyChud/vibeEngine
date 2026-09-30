//! The vibeEngine editor: a Rust-native immediate-mode UI over the engine.
//!
//! The UI is egui rather than a C++ toolkit, for one concrete reason: egui's
//! output is a flat list of coloured triangles in screen space, which is the
//! same shape as the engine's own 2D batcher. So the editor needs no second
//! renderer, no second windowing path and no C++ toolchain — it hands its
//! triangles to `vibe_render`'s UI pipeline and they go out in the frame's
//! draw call like anything else.
//!
//! The crate is `unsafe_code = "forbid"` and holds no Vulkan handles: it decides
//! what to draw and where, and `vibe_render` decides how.

pub mod camera;
pub mod content;
pub mod frame;
pub mod gizmo;
pub mod hierarchy;
pub mod inspector;
pub mod logs;
pub mod session;
pub mod ui_bridge;

pub use camera::{Camera2DController, CameraInput, EditorCamera};
pub use content::{ContentBrowser, ContentEntry, ContentKind};
pub use frame::{Editor, EditorFrame, Panel};
pub use gizmo::{GizmoAxis, GizmoDrag};
pub use hierarchy::{HierarchyPanel, Selection};
pub use inspector::{FieldEdit, FieldValue, InspectorPanel};
pub use logs::{LogLevel, LogPanel, LogSink};
pub use session::{EditorMode, EditorSession, FileAction, SessionError};
pub use ui_bridge::UiDrawData;
pub use vibe_render::ui::UiVertex;
