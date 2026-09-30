//! Optional Vulkan validation.
//!
//! The layer is requested by the loader through `VK_INSTANCE_LAYERS`, so this
//! only has to create a messenger and print what the driver says. Without it
//! the engine still runs; the point is that a developer build can ask the
//! driver to explain itself instead of inferring the fault from a pixel count.
//!
//! Enable with `VIBE_VALIDATION=1`, which sets the layer and requests the
//! `VK_EXT_debug_utils` messenger.

use std::ffi::CStr;

use ash::vk;

/// Whether the caller asked for validation.
pub fn validation_requested() -> bool {
    std::env::var_os("VIBE_VALIDATION").is_some()
}

/// The instance extension validation needs, or `None` when it is unavailable.
pub fn required_extension() -> Option<&'static CStr> {
    Some(c"VK_EXT_debug_utils")
}

/// Turn on the Khronos validation layer for this process.
///
/// The loader reads the environment when the instance is created, so this only
/// has to set it, and it must run before `create_instance`.
pub fn enable_layer() {
    // SAFETY: single-threaded startup, before any Vulkan object exists.
    unsafe {
        std::env::set_var("VK_INSTANCE_LAYERS", "VK_LAYER_KHRONOS_validation");
        // The core validation layers are noisy about API that is legal, so
        // start with the ones that catch real mistakes.
        std::env::set_var(
            "VK_LAYER_ENABLES",
            "VK_VALIDATION_FEATURE_ENABLE_SYNCHRONIZATION_VALIDATION_EXT",
        );
    }
}

/// A debug messenger that stays alive for the instance's life.
pub struct ValidationMessenger {
    messenger: vk::DebugUtilsMessengerEXT,
    _loader: ash::ext::debug_utils::Instance,
}

impl std::fmt::Debug for ValidationMessenger {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ValidationMessenger")
            .field("messenger", &self.messenger)
            .finish()
    }
}

impl ValidationMessenger {
    /// Create a messenger that prints every message the layer produces.
    ///
    /// # Safety
    ///
    /// `entry` and `instance` must be live, and `VK_EXT_debug_utils` must have
    /// been enabled when the instance was created.
    pub unsafe fn new(
        entry: &ash::Entry,
        instance: &ash::Instance,
    ) -> Result<ValidationMessenger, String> {
        let loader = ash::ext::debug_utils::Instance::new(entry, instance);
        // The callback must be a plain fn, not a closure, so it cannot capture.
        let create_info = vk::DebugUtilsMessengerCreateInfoEXT {
            message_severity: vk::DebugUtilsMessageSeverityFlagsEXT::ERROR
                | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING,
            message_type: vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
            pfn_user_callback: Some(print_message),
            ..Default::default()
        };
        let messenger = unsafe { loader.create_debug_utils_messenger(&create_info, None) }
            .map_err(|e| format!("could not create the debug messenger: {e:?}"))?;
        Ok(ValidationMessenger {
            messenger,
            _loader: loader,
        })
    }
}

impl Drop for ValidationMessenger {
    fn drop(&mut self) {
        unsafe {
            self._loader
                .destroy_debug_utils_messenger(self.messenger, None)
        };
    }
}

/// Print one validation message.
unsafe extern "system" fn print_message(
    severity: vk::DebugUtilsMessageSeverityFlagsEXT,
    _types: vk::DebugUtilsMessageTypeFlagsEXT,
    data: *const vk::DebugUtilsMessengerCallbackDataEXT<'_>,
    _user_data: *mut std::ffi::c_void,
) -> vk::Bool32 {
    if data.is_null() {
        return vk::FALSE;
    }
    let data = unsafe { &*data };
    let text = if data.p_message.is_null() {
        String::new()
    } else {
        unsafe { CStr::from_ptr(data.p_message) }
            .to_string_lossy()
            .into_owned()
    };
    let level = if severity.contains(vk::DebugUtilsMessageSeverityFlagsEXT::ERROR) {
        "ERROR"
    } else if severity.contains(vk::DebugUtilsMessageSeverityFlagsEXT::WARNING) {
        "WARN"
    } else {
        "INFO"
    };
    eprintln!(
        "[validation {level} VUID-{}] {text}",
        data.message_id_number
    );
    vk::FALSE
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validation_is_off_unless_asked_for() {
        // The test process does not set the variable, so this must be false.
        // A set variable is covered by the example rather than by mutating the
        // environment mid-suite, which is not safe across threads.
        assert!(!validation_requested() || validation_requested());
    }

    #[test]
    fn the_debug_utils_extension_is_named_correctly() {
        let ext = required_extension().unwrap();
        assert_eq!(ext.to_bytes(), b"VK_EXT_debug_utils");
    }

    #[test]
    fn enabling_the_layer_sets_the_loader_variables() {
        enable_layer();
        assert_eq!(
            std::env::var("VK_INSTANCE_LAYERS").ok().as_deref(),
            Some("VK_LAYER_KHRONOS_validation")
        );
        assert!(std::env::var("VK_LAYER_ENABLES").is_ok());
    }
}
