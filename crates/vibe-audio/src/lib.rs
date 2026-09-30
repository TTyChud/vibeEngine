//! Positional audio: 2D and 3D placement, a listener, volume, and looping.
//!
//! Backed by cpal rather than miniaudio. The `miniaudio` crate does not build
//! on this machine: its bundled bindgen 0.54 panics parsing the system glibc
//! headers. cpal is pure Rust over ALSA/PulseAudio and exposes the same
//! placement API this crate needs.

pub mod error;
pub mod math;
pub mod mixer;
pub mod voice;

pub use error::AudioError;
pub use math::{
    Attenuation, AudioListener, Vec3, clamp_distance, distance_gain, pan_for_x, pitch_for_distance,
};
pub use mixer::{Mixer, TrackId};
pub use voice::{LoopMode, Voice, VoiceState};
