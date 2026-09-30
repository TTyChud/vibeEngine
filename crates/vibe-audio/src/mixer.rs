//! The mixer: owns the output device and sums the voices.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

use crate::error::AudioError;
use crate::math::{Attenuation, AudioListener, Vec3};
use crate::voice::{LoopMode, Voice};

/// A handle to a playing sound.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TrackId(u64);

impl TrackId {
    /// The raw value, for logging.
    pub fn value(self) -> u64 {
        self.0
    }
}

/// The voices and the listener, shared with the audio thread.
#[derive(Debug, Default)]
struct MixerState {
    voices: HashMap<u64, Voice>,
    listener: AudioListener,
}

impl MixerState {
    /// Fill `out` with every voice, summed.
    fn mix(&mut self, out: &mut [f32]) {
        let listener = self.listener;
        for v in out.iter_mut() {
            *v = 0.0;
        }
        let mut finished = Vec::new();
        for (id, voice) in self.voices.iter_mut() {
            if !voice.fill(out, &listener) && voice.is_finished() {
                finished.push(*id);
            }
        }
        for id in finished {
            self.voices.remove(&id);
        }
    }
}

/// Owns the output device and plays voices through it.
///
/// The voices live behind a mutex so the game thread can start and stop sounds
/// while the audio callback reads them. The callback only locks for the length
/// of one block, which is what keeps it from glitching.
pub struct Mixer {
    state: Arc<Mutex<MixerState>>,
    next_id: AtomicU64,
    stream: Option<cpal::Stream>,
    device_name: Option<String>,
}

impl std::fmt::Debug for Mixer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Mixer")
            .field("device", &self.device_name)
            .field(
                "playing",
                &self.state.lock().map(|s| s.voices.len()).unwrap_or(0),
            )
            .finish()
    }
}

impl Mixer {
    /// A mixer with no device open, for tests and headless use.
    pub fn silent() -> Mixer {
        Mixer {
            state: Arc::new(Mutex::new(MixerState::default())),
            next_id: AtomicU64::new(1),
            stream: None,
            device_name: None,
        }
    }

    /// Open the default output device and start mixing.
    ///
    /// # Errors
    ///
    /// Returns [`AudioError::NoOutputDevice`] when the machine has no output,
    /// which is the case on a headless box.
    pub fn open_default() -> Result<Mixer, AudioError> {
        let host = cpal::default_host();
        let device = host
            .default_output_device()
            .ok_or(AudioError::NoOutputDevice)?;
        let device_name = format!("{device:?}");
        let config = device
            .default_output_config()
            .map_err(|e| AudioError::UnsupportedConfig(e.to_string()))?;

        let mixer = Mixer::silent();
        let state = Arc::clone(&mixer.state);
        let stream_config = cpal::StreamConfig {
            channels: config.channels(),
            sample_rate: config.sample_rate(),
            buffer_size: cpal::BufferSize::Default,
        };

        let err_fn = |err| log::warn!("audio stream error: {err}");
        let stream = device
            .build_output_stream(
                stream_config,
                move |data: &mut [f32], _: &cpal::OutputCallbackInfo| {
                    if let Ok(mut s) = state.lock() {
                        s.mix(data);
                    } else {
                        for sample in data.iter_mut() {
                            *sample = 0.0;
                        }
                    }
                },
                err_fn,
                None,
            )
            .map_err(|e| AudioError::OpenDevice(e.to_string()))?;

        stream
            .play()
            .map_err(|e| AudioError::OpenDevice(e.to_string()))?;

        log::info!("audio output open: {device_name}");
        let mut mixer = mixer;
        mixer.stream = Some(stream);
        mixer.device_name = Some(device_name);
        Ok(mixer)
    }

    /// The output device's description, if one is open.
    pub fn device_name(&self) -> Option<&str> {
        self.device_name.as_deref()
    }

    /// Start a sound, returning a handle to stop it later.
    pub fn play(&self, samples: Vec<f32>, loop_mode: LoopMode) -> Result<TrackId, AudioError> {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        let mut voice = Voice::new();
        voice.load(samples, loop_mode)?;
        voice.play()?;
        match self.state.lock() {
            Ok(mut s) => {
                s.voices.insert(id, voice);
                Ok(TrackId(id))
            }
            Err(_) => Err(AudioError::UnsupportedConfig("mixer is poisoned".into())),
        }
    }

    /// Stop a sound and forget it.
    pub fn stop(&self, id: TrackId) -> bool {
        match self.state.lock() {
            Ok(mut s) => s.voices.remove(&id.0).is_some(),
            Err(_) => false,
        }
    }

    /// Whether a sound is still loaded.
    pub fn is_playing(&self, id: TrackId) -> bool {
        self.state
            .lock()
            .map(|s| s.voices.get(&id.0).is_some_and(|v| v.is_playing()))
            .unwrap_or(false)
    }

    /// How many sounds are loaded.
    pub fn voice_count(&self) -> usize {
        self.state.lock().map(|s| s.voices.len()).unwrap_or(0)
    }

    /// Stop every sound.
    pub fn stop_all(&self) {
        if let Ok(mut s) = self.state.lock() {
            s.voices.clear();
        }
    }

    /// Move the listener.
    pub fn set_listener(&self, listener: AudioListener) {
        if let Ok(mut s) = self.state.lock() {
            s.listener = listener;
        }
    }

    /// Place a playing sound.
    pub fn set_position(&self, id: TrackId, position: Vec3) -> bool {
        match self.state.lock() {
            Ok(mut s) => match s.voices.get_mut(&id.0) {
                Some(v) => {
                    v.set_position(position);
                    true
                }
                None => false,
            },
            Err(_) => false,
        }
    }

    /// Set a playing sound's volume.
    pub fn set_volume(&self, id: TrackId, volume: f32) -> bool {
        match self.state.lock() {
            Ok(mut s) => match s.voices.get_mut(&id.0) {
                Some(v) => {
                    v.set_volume(volume);
                    true
                }
                None => false,
            },
            Err(_) => false,
        }
    }

    /// Turn spatial placement on or off for a sound.
    pub fn set_spatial(&self, id: TrackId, spatial: bool) -> bool {
        match self.state.lock() {
            Ok(mut s) => match s.voices.get_mut(&id.0) {
                Some(v) => {
                    v.set_spatial(spatial);
                    true
                }
                None => false,
            },
            Err(_) => false,
        }
    }

    /// Set a sound's distance attenuation.
    pub fn set_attenuation(&self, id: TrackId, attenuation: Attenuation) -> bool {
        match self.state.lock() {
            Ok(mut s) => match s.voices.get_mut(&id.0) {
                Some(v) => {
                    v.set_attenuation(attenuation);
                    true
                }
                None => false,
            },
            Err(_) => false,
        }
    }

    /// Set the master volume.
    pub fn set_master_volume(&self, volume: f32) {
        self.set_listener(AudioListener {
            volume: volume.clamp(0.0, 1.0),
            ..self.listener()
        });
    }

    /// The current listener.
    pub fn listener(&self) -> AudioListener {
        self.state.lock().map(|s| s.listener).unwrap_or_default()
    }

    /// Mix one block by hand, for tests and for headless verification.
    pub fn mix_block(&self, out: &mut [f32]) {
        if let Ok(mut s) = self.state.lock() {
            s.mix(out);
        } else {
            for sample in out.iter_mut() {
                *sample = 0.0;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ramp() -> Vec<f32> {
        vec![0.5f32; 16]
    }

    #[test]
    fn a_silent_mixer_has_no_device_and_no_voices() {
        let m = Mixer::silent();
        assert!(m.device_name().is_none());
        assert_eq!(m.voice_count(), 0);
    }

    #[test]
    fn playing_adds_a_voice() {
        let m = Mixer::silent();
        let id = m.play(ramp(), LoopMode::Loop).unwrap();
        assert_eq!(m.voice_count(), 1);
        assert!(m.is_playing(id));
    }

    #[test]
    fn track_ids_are_unique() {
        let m = Mixer::silent();
        let a = m.play(ramp(), LoopMode::Loop).unwrap();
        let b = m.play(ramp(), LoopMode::Loop).unwrap();
        assert_ne!(a, b);
    }

    #[test]
    fn empty_samples_are_rejected() {
        let m = Mixer::silent();
        assert!(m.play(Vec::new(), LoopMode::Loop).is_err());
        assert_eq!(
            m.voice_count(),
            0,
            "a rejected sound must not be left behind"
        );
    }

    #[test]
    fn stopping_removes_the_voice() {
        let m = Mixer::silent();
        let id = m.play(ramp(), LoopMode::Loop).unwrap();
        assert!(m.stop(id));
        assert_eq!(m.voice_count(), 0);
        assert!(!m.is_playing(id));
    }

    #[test]
    fn stopping_an_unknown_voice_reports_false() {
        let m = Mixer::silent();
        assert!(!m.stop(TrackId(999)));
    }

    #[test]
    fn stop_all_clears_the_mixer() {
        let m = Mixer::silent();
        m.play(ramp(), LoopMode::Loop).unwrap();
        m.play(ramp(), LoopMode::Loop).unwrap();
        m.stop_all();
        assert_eq!(m.voice_count(), 0);
    }

    #[test]
    fn mixing_a_block_produces_the_voice() {
        let m = Mixer::silent();
        m.play(vec![1.0f32; 16], LoopMode::Loop).unwrap();
        let mut out = [0.0f32; 8];
        m.mix_block(&mut out);
        assert!(
            out.iter().any(|s| *s != 0.0),
            "a playing voice must reach the output"
        );
    }

    #[test]
    fn mixing_with_no_voices_silences_the_block() {
        let m = Mixer::silent();
        let mut out = [1.0f32; 8];
        m.mix_block(&mut out);
        assert!(out.iter().all(|s| *s == 0.0));
    }

    #[test]
    fn two_voices_are_summed() {
        let m = Mixer::silent();
        m.play(vec![0.5f32; 16], LoopMode::Loop).unwrap();
        m.play(vec![0.5f32; 16], LoopMode::Loop).unwrap();
        let mut out = [0.0f32; 8];
        m.mix_block(&mut out);
        assert!(
            out[0] > 0.9,
            "two voices at 0.5 should sum past 1.0, got {}",
            out[0]
        );
    }

    #[test]
    fn a_finished_one_shot_is_retired() {
        let m = Mixer::silent();
        // Two stereo frames, consumed in one block.
        m.play(vec![1.0, 1.0, 1.0, 1.0], LoopMode::Once).unwrap();
        let mut out = [0.0f32; 4];
        m.mix_block(&mut out);
        m.mix_block(&mut out);
        assert_eq!(m.voice_count(), 0, "a finished one-shot must be dropped");
    }

    #[test]
    fn moving_the_listener_changes_the_mix() {
        let m = Mixer::silent();
        let id = m.play(vec![1.0f32; 16], LoopMode::Loop).unwrap();
        m.set_spatial(id, true);
        m.set_position(id, Vec3::new(20.0, 0.0, -5.0));

        let mut out = [0.0f32; 8];
        m.mix_block(&mut out);
        let right_source = out[1].abs();

        m.set_listener(AudioListener {
            position: Vec3::new(-40.0, 0.0, 0.0),
            ..AudioListener::at_origin()
        });
        m.mix_block(&mut out);
        let flipped = out[1].abs();
        assert!(
            right_source > flipped,
            "moving the listener changes the pan: {right_source} vs {flipped}"
        );
    }

    #[test]
    fn master_volume_scales_the_mix() {
        let m = Mixer::silent();
        m.play(vec![1.0f32; 16], LoopMode::Loop).unwrap();
        let mut out = [0.0f32; 8];
        m.mix_block(&mut out);
        let loud = out[0].abs();

        m.set_master_volume(0.0);
        m.mix_block(&mut out);
        assert!(
            out[0].abs() < loud * 0.01,
            "master volume 0 must silence the mix"
        );
    }

    #[test]
    fn master_volume_is_clamped() {
        let m = Mixer::silent();
        m.set_master_volume(5.0);
        assert_eq!(m.listener().volume, 1.0);
        m.set_master_volume(-1.0);
        assert_eq!(m.listener().volume, 0.0);
    }

    #[test]
    fn per_voice_volume_works() {
        let m = Mixer::silent();
        let id = m.play(vec![1.0f32; 16], LoopMode::Loop).unwrap();
        assert!(m.set_volume(id, 0.5));
        let mut out = [0.0f32; 8];
        m.mix_block(&mut out);
        assert!((out[0].abs() - 0.5).abs() < 1e-5, "got {}", out[0]);
    }

    #[test]
    fn commands_on_an_unknown_voice_report_false() {
        let m = Mixer::silent();
        let ghost = TrackId(4242);
        assert!(!m.set_volume(ghost, 0.5));
        assert!(!m.set_position(ghost, Vec3::ZERO));
        assert!(!m.set_spatial(ghost, true));
        assert!(!m.set_attenuation(ghost, Attenuation::default()));
    }

    #[test]
    fn many_voices_can_play_at_once() {
        let m = Mixer::silent();
        for _ in 0..32 {
            m.play(ramp(), LoopMode::Loop).unwrap();
        }
        assert_eq!(m.voice_count(), 32);
    }
}
