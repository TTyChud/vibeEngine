//! A single playing sound: its samples, its playback state, and its placement.

use crate::math::{Attenuation, AudioListener, Vec3};

/// What happens when a sound reaches its end.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LoopMode {
    /// Stop at the end.
    #[default]
    Once,
    /// Restart from the beginning.
    Loop,
    /// Play forward, then backward, then forward again.
    PingPong,
}

/// Where a voice is in its lifetime.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum VoiceState {
    /// Created but not yet given samples.
    #[default]
    Empty,
    /// Holding samples, not started.
    Ready,
    /// Playing.
    Playing,
    /// Finished; a one-shot voice stays in this state until stopped.
    Finished,
}

/// One sound being played, with its placement.
///
/// The voice owns the sample data and the cursor into it, so the audio thread
/// can read the next block without touching the game state. Placement is read
/// once per block from the listener.
#[derive(Debug, Clone)]
pub struct Voice {
    /// Interleaved stereo samples in `-1.0..=1.0`.
    samples: Vec<f32>,
    /// Position within the samples, fractional so a rate below 1.0 advances
    /// smoothly instead of rounding to the same frame forever.
    cursor: f32,
    /// Playback rate, where 1.0 is the recorded speed.
    rate: f32,
    /// Volume in `0.0..=1.0`.
    volume: f32,
    /// Where the sound is in the world.
    position: Vec3,
    /// Whether placement is applied at all.
    spatial: bool,
    /// How loud it gets as it moves away.
    attenuation: Attenuation,
    /// What happens at the end.
    loop_mode: LoopMode,
    /// Playback state.
    state: VoiceState,
    /// Whether the last block read went backwards, for ping-pong.
    reversed: bool,
}

impl Default for Voice {
    fn default() -> Self {
        Voice::new()
    }
}

impl Voice {
    /// An empty voice.
    pub fn new() -> Voice {
        Voice {
            samples: Vec::new(),
            cursor: 0.0,
            rate: 1.0,
            volume: 1.0,
            position: Vec3::ZERO,
            spatial: false,
            attenuation: Attenuation::default(),
            loop_mode: LoopMode::Once,
            state: VoiceState::Empty,
            reversed: false,
        }
    }

    /// Load interleaved stereo samples.
    ///
    /// An empty buffer is rejected, because a voice with no samples would
    /// otherwise loop forever on nothing.
    pub fn load(
        &mut self,
        samples: Vec<f32>,
        loop_mode: LoopMode,
    ) -> Result<(), crate::AudioError> {
        if samples.is_empty() {
            return Err(crate::AudioError::UnsupportedConfig(
                "voice needs samples".into(),
            ));
        }
        self.samples = samples;
        self.loop_mode = loop_mode;
        self.cursor = 0.0;
        self.reversed = false;
        self.state = VoiceState::Ready;
        Ok(())
    }

    /// Begin playing.
    pub fn play(&mut self) -> Result<(), crate::AudioError> {
        if self.state == VoiceState::Empty {
            return Err(crate::AudioError::NotStarted(0));
        }
        self.state = VoiceState::Playing;
        Ok(())
    }

    /// Stop and rewind to the start.
    pub fn stop(&mut self) {
        self.cursor = 0.0;
        self.reversed = false;
        self.state = if self.samples.is_empty() {
            VoiceState::Empty
        } else {
            VoiceState::Ready
        };
    }

    /// Current playback state.
    pub fn state(&self) -> VoiceState {
        self.state
    }

    /// Whether the voice is still producing sound.
    pub fn is_playing(&self) -> bool {
        self.state == VoiceState::Playing
    }

    /// The sample cursor, in frames from the start.
    pub fn cursor(&self) -> usize {
        self.cursor as usize
    }

    /// Playback rate, clamped so a voice cannot run backwards or vanish.
    pub fn set_rate(&mut self, rate: f32) {
        if rate.is_finite() && rate > 0.0 {
            self.rate = rate.clamp(0.01, 4.0);
        }
    }

    /// The playback rate.
    pub fn rate(&self) -> f32 {
        self.rate
    }

    /// Volume, clamped to `0.0..=1.0`.
    pub fn set_volume(&mut self, volume: f32) {
        if volume.is_finite() {
            self.volume = volume.clamp(0.0, 1.0);
        }
    }

    /// The volume.
    pub fn volume(&self) -> f32 {
        self.volume
    }

    /// Place the sound in the world and apply distance attenuation.
    pub fn set_position(&mut self, position: Vec3) {
        self.position = position;
    }

    /// Turn spatial placement on or off.
    pub fn set_spatial(&mut self, spatial: bool) {
        self.spatial = spatial;
    }

    /// Whether placement is applied.
    pub fn is_spatial(&self) -> bool {
        self.spatial
    }

    /// The distance attenuation model.
    pub fn set_attenuation(&mut self, attenuation: Attenuation) {
        if attenuation.is_valid() {
            self.attenuation = attenuation;
        }
    }

    /// What happens at the end of the samples.
    pub fn loop_mode(&self) -> LoopMode {
        self.loop_mode
    }

    /// How many samples the voice holds.
    pub fn sample_count(&self) -> usize {
        self.samples.len()
    }

    /// True when the voice has finished and will not produce more sound.
    pub fn is_finished(&self) -> bool {
        self.state == VoiceState::Finished
    }

    /// Fill `out` with the next block of samples, applying volume and pan.
    ///
    /// Returns `false` once a one-shot voice has finished, so the caller can
    /// retire it. A looping voice never returns `false`.
    pub fn fill(&mut self, out: &mut [f32], listener: &AudioListener) -> bool {
        if self.state == VoiceState::Finished || self.samples.is_empty() {
            // Silence the block: the mixer sums voices into it, and a finished
            // voice that writes nothing would leave the previous block's tail
            // audible.
            for s in out.iter_mut() {
                *s = 0.0;
            }
            return false;
        }
        if self.state != VoiceState::Playing {
            return false;
        }

        // A non-spatial sound still follows the listener's volume; it only
        // skips the pan and the distance falloff.
        let (pan, spatial_gain) = if self.spatial {
            listener.spatialise(self.position, self.attenuation)
        } else {
            (0.0, listener.volume)
        };
        let gain = self.volume * spatial_gain;
        // Unity-at-centre pan law: a centred source plays at full volume. A true
        // equal-power law would divide a centred source by sqrt(2), which
        // audibly dips as a source passes the listener.
        let pan = pan.clamp(-1.0, 1.0);
        let left = gain * (1.0 - pan * 0.5);
        let right = gain * (1.0 + pan * 0.5);

        let frames = self.samples.len() / 2;
        if frames == 0 {
            self.state = VoiceState::Finished;
            return false;
        }
        let out_frames = out.len() / 2;
        let step = self.rate.max(0.01) as f32;

        // `self.cursor` is a float position so a fractional rate advances
        // smoothly, and `reversed` is the direction. The end condition depends
        // on the direction, which is what made this tangled before.
        let mut position = self.cursor as f32;
        let mut direction = if self.reversed { -1.0f32 } else { 1.0f32 };
        let mut produced = 0usize;

        while produced < out_frames {
            let past_end = position >= frames as f32;
            let past_start = position < 0.0;

            if past_end || past_start {
                match self.loop_mode {
                    LoopMode::Once => {
                        // Only a forward one-shot can run out; a reversed voice
                        // stopping at the start is the same thing.
                        self.state = VoiceState::Finished;
                        self.cursor = 0.0;
                        // A finished voice contributes nothing, and the
                        // remainder of the block must be silence rather than
                        // left holding whatever was there.
                        for s in out[produced * 2..].iter_mut() {
                            *s = 0.0;
                        }
                        return produced > 0;
                    }
                    LoopMode::Loop => {
                        position = 0.0;
                        direction = 1.0;
                    }
                    LoopMode::PingPong => {
                        // Bounce: reverse direction and step back inside the
                        // buffer, so the endpoints are heard once, not twice.
                        direction = -direction;
                        position = if past_end { frames as f32 - step } else { step };
                    }
                }
            }

            let index = (position as usize).min(frames - 1);
            let base = index * 2;
            out[produced * 2] += self.samples[base] * left;
            out[produced * 2 + 1] += self.samples[base + 1] * right;

            position += direction * step;
            produced += 1;
        }

        self.cursor = position.max(0.0);
        self.reversed = direction < 0.0;
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Four stereo frames: L,R pairs of -1 and 1.
    fn ramp() -> Vec<f32> {
        vec![-1.0, -1.0, -0.5, -0.5, 0.5, 0.5, 1.0, 1.0]
    }

    fn playing(loop_mode: LoopMode) -> Voice {
        let mut v = Voice::new();
        v.load(ramp(), loop_mode).unwrap();
        v.play().unwrap();
        v
    }

    #[test]
    fn a_new_voice_is_empty_and_not_playing() {
        let v = Voice::new();
        assert_eq!(v.state(), VoiceState::Empty);
        assert!(!v.is_playing());
        assert_eq!(v.sample_count(), 0);
    }

    #[test]
    fn loading_samples_readies_the_voice() {
        let mut v = Voice::new();
        v.load(ramp(), LoopMode::Once).unwrap();
        assert_eq!(v.state(), VoiceState::Ready);
        assert_eq!(v.sample_count(), 8);
    }

    #[test]
    fn empty_samples_are_rejected() {
        let mut v = Voice::new();
        assert!(v.load(Vec::new(), LoopMode::Loop).is_err());
        assert_eq!(v.state(), VoiceState::Empty);
    }

    #[test]
    fn a_voice_with_no_samples_cannot_play() {
        let mut v = Voice::new();
        assert!(v.play().is_err());
    }

    #[test]
    fn playing_advances_the_cursor() {
        let mut v = playing(LoopMode::Loop);
        let mut out = [0.0f32; 4];
        let l = AudioListener::at_origin();
        v.fill(&mut out, &l);
        assert!(v.cursor() > 0);
    }

    #[test]
    fn a_one_shot_finishes_and_reports_false() {
        let mut v = playing(LoopMode::Once);
        let l = AudioListener::at_origin();
        let mut out = [0.0f32; 8];
        let mut alive = true;
        // Keep pulling: the first block consumes the samples, the next reports
        // the finish.
        for _ in 0..3 {
            alive = v.fill(&mut out, &l);
            if !alive {
                break;
            }
        }
        assert!(!alive || v.is_finished(), "a one-shot must eventually stop");
    }

    #[test]
    fn a_looping_voice_never_reports_finished() {
        let mut v = playing(LoopMode::Loop);
        let l = AudioListener::at_origin();
        let mut out = [0.0f32; 8];
        for _ in 0..50 {
            assert!(v.fill(&mut out, &l), "a looping voice must keep playing");
        }
        assert!(!v.is_finished());
    }

    #[test]
    fn a_ping_pong_voice_never_reports_finished() {
        let mut v = playing(LoopMode::PingPong);
        let l = AudioListener::at_origin();
        let mut out = [0.0f32; 4];
        for _ in 0..50 {
            assert!(v.fill(&mut out, &l));
        }
        assert!(!v.is_finished());
    }

    #[test]
    fn ping_pong_reverses_direction() {
        // A long buffer, so several blocks pass before the end is reached and
        // the reversal is actually observed rather than wrapped past.
        let long: Vec<f32> = (0..200).map(|i| (i % 16) as f32 / 16.0).collect();
        let mut v = Voice::new();
        v.load(long, LoopMode::PingPong).unwrap();
        v.play().unwrap();

        let l = AudioListener::at_origin();
        let mut out = [0.0f32; 8];
        let mut rose = false;
        for _ in 0..200 {
            let before = v.cursor();
            v.fill(&mut out, &l);
            if v.cursor() < before {
                rose = true;
                break;
            }
        }
        assert!(rose, "ping-pong must eventually play backwards");
    }

    #[test]
    fn a_stopped_voice_rewinds_and_is_not_playing() {
        let mut v = playing(LoopMode::Loop);
        let l = AudioListener::at_origin();
        let mut out = [0.0f32; 4];
        v.fill(&mut out, &l);
        v.stop();
        assert_eq!(v.cursor(), 0);
        assert!(!v.is_playing());
    }

    #[test]
    fn volume_zero_silences_the_voice() {
        let mut v = playing(LoopMode::Loop);
        v.set_volume(0.0);
        let l = AudioListener::at_origin();
        // A zeroed block: fill sums into it, so a pre-filled one would keep
        // whatever was already there.
        let mut out = [0.0f32; 8];
        v.fill(&mut out, &l);
        assert!(
            out.iter().all(|s| *s == 0.0),
            "zero volume must produce silence"
        );
    }

    #[test]
    fn volume_is_clamped() {
        let mut v = Voice::new();
        v.set_volume(5.0);
        assert_eq!(v.volume(), 1.0);
        v.set_volume(-2.0);
        assert_eq!(v.volume(), 0.0);
        v.set_volume(f32::NAN);
        assert_eq!(
            v.volume(),
            0.0,
            "a nan volume is ignored, leaving the last value"
        );
    }

    #[test]
    fn rate_is_clamped_to_a_usable_range() {
        let mut v = Voice::new();
        v.set_rate(0.0);
        assert!(v.rate() > 0.0, "a zero rate would stall the cursor");
        v.set_rate(100.0);
        assert!(v.rate() <= 4.0);
        v.set_rate(f32::NAN);
        assert!(v.rate() > 0.0, "a nan rate is ignored");
    }

    #[test]
    fn a_faster_rate_covers_more_samples() {
        // A buffer longer than one block, so a looping voice does not wrap and
        // land back on the same cursor whatever its rate.
        let long: Vec<f32> = (0..64).map(|i| (i % 16) as f32 / 16.0).collect();
        let l = AudioListener::at_origin();

        let mut slow = Voice::new();
        slow.load(long.clone(), LoopMode::Loop).unwrap();
        slow.play().unwrap();
        let mut fast = Voice::new();
        fast.load(long, LoopMode::Loop).unwrap();
        fast.play().unwrap();
        fast.set_rate(4.0);

        // fill sums into the block, as the mixer does, so each voice gets a
        // zeroed block rather than the other's output.
        let mut out = [0.0f32; 8];
        slow.fill(&mut out, &l);
        let mut second = [0.0f32; 8];
        fast.fill(&mut second, &l);
        assert!(
            fast.cursor() > slow.cursor(),
            "a faster voice advances further"
        );
    }

    #[test]
    fn a_centred_voice_copies_the_samples() {
        let mut v = playing(LoopMode::Loop);
        let l = AudioListener::at_origin();
        let mut out = [0.0f32; 8];
        v.fill(&mut out, &l);
        // Centred, so the first two output samples are the first two input
        // samples, unchanged.
        assert!((out[0] - -1.0).abs() < 1e-6, "got {}", out[0]);
        assert!((out[1] - -1.0).abs() < 1e-6);
    }

    #[test]
    fn a_spatial_voice_to_the_right_is_louder_on_the_right() {
        let mut v = playing(LoopMode::Loop);
        v.set_spatial(true);
        v.set_position(Vec3::new(20.0, 0.0, -5.0));
        let l = AudioListener::at_origin();
        let mut out = [0.0f32; 8];
        v.fill(&mut out, &l);
        let left = out[0].abs();
        let right = out[1].abs();
        assert!(
            right > left,
            "a right-hand source must be louder on the right: {left} vs {right}"
        );
    }

    #[test]
    fn a_distant_spatial_voice_is_quieter_than_a_near_one() {
        let l = AudioListener::at_origin();
        let mut near = playing(LoopMode::Loop);
        let mut far = playing(LoopMode::Loop);
        near.set_spatial(true);
        far.set_spatial(true);
        near.set_position(Vec3::new(0.0, 0.0, -1.0));
        far.set_position(Vec3::new(0.0, 0.0, -80.0));

        let mut out = [0.0f32; 8];
        near.fill(&mut out, &l);
        let near_level = out[0].abs();
        let mut second = [0.0f32; 8];
        far.fill(&mut second, &l);
        let far_level = second[0].abs();
        assert!(
            near_level > far_level,
            "near {near_level} should exceed far {far_level}"
        );
    }

    #[test]
    fn a_nonspatial_voice_ignores_its_position() {
        let mut a = playing(LoopMode::Loop);
        let mut b = playing(LoopMode::Loop);
        b.set_position(Vec3::new(100.0, 0.0, 0.0));
        let l = AudioListener::at_origin();
        // Two zeroed blocks: fill sums into whatever it is given, so reusing
        // the first would double it.
        let mut first = [0.0f32; 8];
        a.fill(&mut first, &l);
        let mut second = [0.0f32; 8];
        b.fill(&mut second, &l);
        assert_eq!(
            first, second,
            "without spatial placement the position is irrelevant"
        );
    }

    #[test]
    fn an_invalid_attenuation_is_ignored() {
        let mut v = Voice::new();
        v.set_attenuation(Attenuation::new(10.0, 5.0));
        assert_eq!(
            v.attenuation,
            Attenuation::default(),
            "a rejected model leaves the default"
        );
    }

    #[test]
    fn filling_a_voice_that_is_not_playing_writes_nothing() {
        let mut v = Voice::new();
        v.load(ramp(), LoopMode::Loop).unwrap();
        let l = AudioListener::at_origin();
        let mut out = [1.0f32; 8];
        assert!(!v.fill(&mut out, &l));
        assert!(
            out.iter().all(|s| *s == 1.0),
            "an unstarted voice must not write samples"
        );
    }

    #[test]
    fn the_finished_block_is_silenced_rather_than_left_stale() {
        let mut v = Voice::new();
        // One stereo frame only, so the second block runs off the end.
        v.load(vec![0.5, 0.5], LoopMode::Once).unwrap();
        v.play().unwrap();
        let l = AudioListener::at_origin();
        let mut out = [9.0f32; 4];
        v.fill(&mut out, &l);
        v.fill(&mut out, &l);
        assert!(
            out.iter().all(|s| *s == 0.0),
            "a finished voice must write silence, not stale samples"
        );
    }

    #[test]
    fn an_odd_length_buffer_does_not_panic() {
        let mut v = Voice::new();
        v.load(vec![0.5; 7], LoopMode::Loop).unwrap();
        v.play().unwrap();
        let l = AudioListener::at_origin();
        let mut out = [0.0f32; 6];
        assert!(v.fill(&mut out, &l));
    }
}
