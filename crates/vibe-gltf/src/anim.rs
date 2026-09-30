//! Skeletal animation: keyframe sampling, clip playback and skinning.
//!
//! glTF stores each channel as a sorted list of (time, value) pairs and leaves
//! interpolation to the sampler, of which only LINEAR and STEP are worth
//! supporting. Everything here is arithmetic on those pairs, so it is tested
//! without a document: the interesting failures are all at the boundaries, and
//! a clip that overshoots its end is the one that shows up as a mesh snapping.

use glam::{Mat4, Quat, Vec3, Vec4};

use crate::error::GltfError;
use crate::mesh::Skin;

/// How a channel interpolates between keyframes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Interpolation {
    /// Hold the earlier value until the next keyframe's time.
    Step,
    /// Interpolate between the two values.
    #[default]
    Linear,
    /// A cubic spline. Read as linear: the control points are not reconstructed,
    /// so the motion is smooth but not the one the exporter authored.
    Cubic,
}

/// The value a channel animates.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ChannelValue {
    /// A translation, in metres.
    Translation(Vec3),
    /// A rotation.
    Rotation(Quat),
    /// A scale factor.
    Scale(Vec3),
    /// A scalar weight, for morph targets.
    Weight(f32),
}

impl ChannelValue {
    /// The value as four floats, which is how a matrix column is filled.
    pub fn to_vec4(self) -> Vec4 {
        match self {
            ChannelValue::Translation(v) | ChannelValue::Scale(v) => v.extend(1.0),
            // A quaternion is xyzw, and the matrix wants the same four numbers
            // in the same order, so this is a straight widening.
            ChannelValue::Rotation(q) => Vec4::new(q.x, q.y, q.z, q.w),
            ChannelValue::Weight(w) => Vec4::new(w, 0.0, 0.0, 0.0),
        }
    }

    /// Linear interpolation between two values of the same kind.
    ///
    /// The rotation branch slerps; the rest lerp componentwise. Mixing a
    /// rotation with a translation is a malformed document, so the translation
    /// is returned rather than panicking.
    pub fn lerp(self, other: ChannelValue, t: f32) -> ChannelValue {
        match (self, other) {
            (ChannelValue::Rotation(a), ChannelValue::Rotation(b)) => {
                ChannelValue::Rotation(a.slerp(b, t))
            }
            (ChannelValue::Translation(a), ChannelValue::Translation(b)) => {
                ChannelValue::Translation(a.lerp(b, t))
            }
            (ChannelValue::Scale(a), ChannelValue::Scale(b)) => ChannelValue::Scale(a.lerp(b, t)),
            (ChannelValue::Weight(a), ChannelValue::Weight(b)) => {
                ChannelValue::Weight(a + (b - a) * t)
            }
            (a, _) => a,
        }
    }
}

/// One keyframe.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Keyframe {
    /// Seconds from the start of the clip.
    pub time: f32,
    /// The value at that time.
    pub value: ChannelValue,
}

/// One animated property: a target and its keyframes, sorted by time.
#[derive(Debug, Clone, PartialEq)]
pub struct Channel {
    /// Index of the node this channel drives.
    pub node: usize,
    /// Which property of that node.
    pub path: ChannelPath,
    /// The keyframes, in increasing time order.
    pub keyframes: Vec<Keyframe>,
}

impl Channel {
    /// The value at a time, holding the last value past the end.
    ///
    /// Holding rather than clamping to zero matters: a translation channel that
    /// returns the origin past its last keyframe snaps the mesh to the root as
    /// soon as the clock runs over.
    pub fn sample(&self, time: f32) -> ChannelValue {
        let Some(first) = self.keyframes.first() else {
            return ChannelValue::Weight(0.0);
        };
        if time <= first.time {
            return first.value;
        }
        let Some(last) = self.keyframes.last() else {
            return first.value;
        };
        if time >= last.time {
            return last.value;
        }
        // Find the bracketing pair. The list is short, so a linear scan beats
        // a binary search's setup and both are far below frame budget.
        for pair in self.keyframes.windows(2) {
            let (a, b) = (&pair[0], &pair[1]);
            if time >= a.time && time <= b.time {
                let span = b.time - a.time;
                let t = if span <= f32::EPSILON {
                    0.0
                } else {
                    (time - a.time) / span
                };
                return a.value.lerp(b.value, t);
            }
        }
        last.value
    }

    /// The time of the last keyframe, which is where the clip really ends.
    pub fn duration(&self) -> f32 {
        self.keyframes.last().map(|k| k.time).unwrap_or(0.0)
    }

    /// The time of the first keyframe, which need not be zero.
    pub fn start(&self) -> f32 {
        self.keyframes.first().map(|k| k.time).unwrap_or(0.0)
    }

    /// Check the keyframe times are usable.
    ///
    /// A descending pair would make the interpolation divide by a negative
    /// span and extrapolate backwards, which looks like a mesh exploding rather
    /// than like a bad file.
    pub fn validate(&self, animation: usize, sampler: usize) -> Result<(), GltfError> {
        let mut previous = f32::NEG_INFINITY;
        for k in &self.keyframes {
            if !k.time.is_finite() {
                return Err(GltfError::UnsupportedPath {
                    animation,
                    sampler,
                    path: format!("keyframe time {} is not finite", k.time),
                });
            }
            if k.time < previous {
                return Err(GltfError::UnsupportedPath {
                    animation,
                    sampler,
                    path: format!(
                        "keyframe times must not descend: {} follows {}",
                        k.time, previous
                    ),
                });
            }
            previous = k.time;
        }
        Ok(())
    }
}

/// Which property of a node a channel drives.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ChannelPath {
    /// `translation`
    Translation,
    /// `rotation`
    Rotation,
    /// `scale`
    Scale,
    /// `weights`, for morph targets
    Weights,
}

impl ChannelPath {
    /// The glTF name for this path.
    pub fn gltf_name(self) -> &'static str {
        match self {
            ChannelPath::Translation => "translation",
            ChannelPath::Rotation => "rotation",
            ChannelPath::Scale => "scale",
            ChannelPath::Weights => "weights",
        }
    }

    /// Parse a glTF channel path.
    pub fn parse(name: &str) -> Option<ChannelPath> {
        match name {
            "translation" => Some(ChannelPath::Translation),
            "rotation" => Some(ChannelPath::Rotation),
            "scale" => Some(ChannelPath::Scale),
            "weights" => Some(ChannelPath::Weights),
            _ => None,
        }
    }
}

/// How a clip repeats.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LoopMode {
    /// Stop at the end.
    #[default]
    Once,
    /// Restart from the beginning.
    Loop,
    /// Play forwards then backwards.
    PingPong,
}

/// A named animation clip.
#[derive(Debug, Clone, PartialEq)]
pub struct Clip {
    /// The clip's name, from the document or a default.
    pub name: String,
    /// The channels, in whatever order the document had them.
    pub channels: Vec<Channel>,
}

impl Clip {
    /// How long the clip runs, in seconds.
    ///
    /// This is the longest channel, not the shortest: a clip cut to the
    /// shortest channel drops the limb back to its bind pose partway through,
    /// which is more visible than running slightly long.
    pub fn duration(&self) -> f32 {
        self.channels
            .iter()
            .map(|c| c.duration())
            .fold(0.0, f32::max)
    }

    /// When the clip's first keyframe lands, for a clip that starts late.
    pub fn start(&self) -> f32 {
        self.channels
            .iter()
            .map(|c| c.start())
            .fold(f32::INFINITY, f32::min)
    }

    /// The value of every channel at a time, paired with the node it drives.
    pub fn sample(&self, time: f32) -> Vec<(usize, ChannelValue)> {
        self.channels
            .iter()
            .map(|c| (c.node, c.sample(time)))
            .collect()
    }

    /// Map a playback time through the loop mode.
    ///
    /// A zero-length clip has no period to divide by, so it returns the time
    /// unchanged rather than producing a NaN that then propagates into every
    /// bone matrix.
    pub fn wrap(&self, time: f32, mode: LoopMode) -> f32 {
        let duration = self.duration();
        if duration <= f32::EPSILON {
            return time;
        }
        let start = if self.start().is_finite() {
            self.start()
        } else {
            0.0
        };
        let span = duration - start;
        if span <= f32::EPSILON {
            return start;
        }
        let local = time - start;
        match mode {
            LoopMode::Once => local.clamp(0.0, span),
            LoopMode::Loop => local.rem_euclid(span),
            LoopMode::PingPong => {
                // A triangle wave: fold the time into 0..span, then reflect it.
                let folded = local.rem_euclid(span * 2.0);
                if folded > span {
                    span * 2.0 - folded
                } else {
                    folded
                }
            }
        }
    }

    /// Check every channel's keyframes.
    pub fn validate(&self, animation: usize) -> Result<(), GltfError> {
        for (sampler, channel) in self.channels.iter().enumerate() {
            channel.validate(animation, sampler)?;
        }
        Ok(())
    }
}

/// A playing clip, with its clock.
#[derive(Debug, Clone, PartialEq)]
pub struct Playback {
    /// Which clip is playing.
    pub clip_index: usize,
    /// Seconds into the clip.
    pub time: f32,
    /// Playback rate; negative plays backwards.
    pub speed: f32,
    /// How the clip repeats.
    pub loop_mode: LoopMode,
    /// True once a `Once` clip has run past its end.
    pub finished: bool,
}

impl Playback {
    /// A playback at the start of a clip, at normal speed.
    pub fn new(clip_index: usize) -> Playback {
        Playback {
            clip_index,
            time: 0.0,
            speed: 1.0,
            loop_mode: LoopMode::Loop,
            finished: false,
        }
    }

    /// Advance the clock, wrapping or stopping as the loop mode says.
    ///
    /// Returns the time to sample at, which is the wrapped time for the clip's
    /// own loop mode. The caller's `Once` stop is separate: a clip can loop
    /// internally while the caller is still deciding when to stop.
    pub fn advance(&mut self, clip: &Clip, delta: f32) -> f32 {
        self.time += delta * self.speed;
        let duration = clip.duration();
        if duration <= f32::EPSILON {
            return self.time;
        }
        if self.loop_mode == LoopMode::Once && self.time >= duration {
            self.time = duration;
            self.finished = true;
            return duration;
        }
        clip.wrap(self.time, self.loop_mode)
    }

    /// Seek to an absolute time.
    pub fn seek(&mut self, time: f32) {
        self.time = time;
        self.finished = false;
    }

    /// True when a `Once` clip has reached its end.
    pub fn is_finished(&self) -> bool {
        self.finished
    }
}

/// The bone matrices for a frame: one `mat4` per joint of the skin.
pub fn bone_matrices(skin: &Skin, world: &[Mat4]) -> Vec<Mat4> {
    crate::mesh::skin_matrices(skin, world)
}

/// Compose a node's local matrix from a channel's sampled value.
pub fn compose_local(path: ChannelPath, value: ChannelValue, base: glam::Affine3A) -> Mat4 {
    // The channel's value is the animated part of the node's own transform, so
    // it is composed on the left of the rest of the node's local transform.
    let base: Mat4 = base.into();
    match path {
        ChannelPath::Translation => {
            let t = match value {
                ChannelValue::Translation(v) => v,
                _ => Vec3::ZERO,
            };
            Mat4::from_translation(t) * base
        }
        ChannelPath::Rotation => {
            let r = match value {
                ChannelValue::Rotation(q) => q,
                _ => Quat::IDENTITY,
            };
            Mat4::from_quat(r) * base
        }
        ChannelPath::Scale => {
            let s = match value {
                ChannelValue::Scale(v) => v,
                _ => Vec3::ONE,
            };
            Mat4::from_scale(s) * base
        }
        ChannelPath::Weights => base,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(time: f32, value: ChannelValue) -> Keyframe {
        Keyframe { time, value }
    }

    fn translation_channel(node: usize, times: &[f32], values: &[f32]) -> Channel {
        Channel {
            node,
            path: ChannelPath::Translation,
            keyframes: times
                .iter()
                .zip(values.iter())
                .map(|(t, v)| key(*t, ChannelValue::Translation(Vec3::new(*v, 0.0, 0.0))))
                .collect(),
        }
    }

    fn rotation_channel(node: usize) -> Channel {
        Channel {
            node,
            path: ChannelPath::Rotation,
            keyframes: vec![
                key(0.0, ChannelValue::Rotation(Quat::IDENTITY)),
                key(
                    1.0,
                    ChannelValue::Rotation(Quat::from_rotation_z(std::f32::consts::FRAC_PI_2)),
                ),
            ],
        }
    }

    fn clip_of(channels: Vec<Channel>) -> Clip {
        Clip {
            name: "test".to_string(),
            channels,
        }
    }

    #[test]
    fn sampling_at_the_first_keyframe_returns_it() {
        let c = translation_channel(0, &[0.0, 1.0], &[0.0, 10.0]);
        match c.sample(0.0) {
            ChannelValue::Translation(v) => assert_eq!(v.x, 0.0),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn sampling_between_keyframes_interpolates() {
        let c = translation_channel(0, &[0.0, 1.0], &[0.0, 10.0]);
        match c.sample(0.5) {
            ChannelValue::Translation(v) => assert!((v.x - 5.0).abs() < 1e-5, "{v:?}"),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn sampling_at_the_last_keyframe_returns_it() {
        let c = translation_channel(0, &[0.0, 1.0], &[0.0, 10.0]);
        match c.sample(1.0) {
            ChannelValue::Translation(v) => assert!((v.x - 10.0).abs() < 1e-5, "{v:?}"),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn sampling_past_the_end_holds_the_last_value() {
        // The bug this prevents: a channel that returns the origin past its
        // last keyframe snaps the mesh to the root as the clock runs over.
        let c = translation_channel(0, &[0.0, 1.0], &[3.0, 10.0]);
        match c.sample(99.0) {
            ChannelValue::Translation(v) => assert!((v.x - 10.0).abs() < 1e-5, "{v:?}"),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn sampling_before_the_start_holds_the_first_value() {
        let c = translation_channel(0, &[2.0, 3.0], &[7.0, 9.0]);
        match c.sample(0.0) {
            ChannelValue::Translation(v) => assert!((v.x - 7.0).abs() < 1e-5, "{v:?}"),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn sampling_an_empty_channel_is_finite() {
        let c = Channel {
            node: 0,
            path: ChannelPath::Translation,
            keyframes: Vec::new(),
        };
        match c.sample(1.0) {
            ChannelValue::Weight(w) => assert!(w.is_finite()),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn a_single_keyframe_holds() {
        let c = translation_channel(0, &[5.0], &[42.0]);
        for t in [0.0, 5.0, 100.0] {
            match c.sample(t) {
                ChannelValue::Translation(v) => assert!((v.x - 42.0).abs() < 1e-5, "t={t}"),
                other => panic!("{other:?}"),
            }
        }
    }

    #[test]
    fn two_keyframes_at_the_same_time_do_not_divide_by_zero() {
        // A zero span would make the interpolation t = 0/0, which is NaN and
        // then propagates into the vertex.
        let c = translation_channel(0, &[1.0, 1.0], &[0.0, 10.0]);
        match c.sample(1.0) {
            ChannelValue::Translation(v) => assert!(v.x.is_finite(), "{v:?}"),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn rotations_slerp_rather_than_lerp() {
        let c = rotation_channel(0);
        match c.sample(0.5) {
            ChannelValue::Rotation(q) => {
                // Halfway through a quarter turn is an eighth turn, 22.5
                // degrees. glam's to_axis_angle reports the full angle, not the
                // half angle its name suggests the value is.
                let (axis, angle) = q.to_axis_angle();
                assert!(
                    (angle.abs() - std::f32::consts::FRAC_PI_4).abs() < 1e-4,
                    "got {angle} about {axis:?}"
                );
                assert!((axis - Vec3::Z).length() < 1e-6, "{axis:?}");
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn a_rotation_holds_its_arc_length() {
        // A componentwise lerp of two unit quaternions is shorter than one, so
        // a slerped rotation must keep its norm.
        let c = rotation_channel(0);
        match c.sample(0.3) {
            ChannelValue::Rotation(q) => assert!((q.length() - 1.0).abs() < 1e-4, "{q:?}"),
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn interpolating_across_kinds_keeps_the_first_value() {
        // A malformed document pairing a translation with a rotation must not
        // panic in the middle of a frame.
        let out =
            ChannelValue::Translation(Vec3::X).lerp(ChannelValue::Rotation(Quat::IDENTITY), 0.5);
        assert_eq!(out, ChannelValue::Translation(Vec3::X));
    }

    #[test]
    fn a_weight_channel_interpolates() {
        let out = ChannelValue::Weight(0.0).lerp(ChannelValue::Weight(1.0), 0.25);
        assert_eq!(out, ChannelValue::Weight(0.25));
    }

    #[test]
    fn a_value_converts_to_four_components() {
        assert_eq!(
            ChannelValue::Weight(2.0).to_vec4(),
            Vec4::new(2.0, 0.0, 0.0, 0.0)
        );
        assert_eq!(
            ChannelValue::Scale(Vec3::ONE).to_vec4(),
            Vec4::new(1.0, 1.0, 1.0, 1.0)
        );
    }

    #[test]
    fn a_channel_reports_its_span() {
        let c = translation_channel(0, &[0.5, 2.0], &[0.0, 1.0]);
        assert!((c.start() - 0.5).abs() < 1e-6);
        assert!((c.duration() - 2.0).abs() < 1e-6);
    }

    #[test]
    fn an_empty_channel_has_no_span() {
        let c = Channel {
            node: 0,
            path: ChannelPath::Scale,
            keyframes: Vec::new(),
        };
        assert_eq!(c.duration(), 0.0);
        assert_eq!(c.start(), 0.0);
    }

    #[test]
    fn ascending_keyframes_validate() {
        let c = translation_channel(0, &[0.0, 1.0, 2.0], &[0.0, 1.0, 2.0]);
        assert!(c.validate(0, 0).is_ok());
    }

    #[test]
    fn descending_keyframes_are_rejected() {
        let c = translation_channel(0, &[0.0, 2.0, 1.0], &[0.0, 1.0, 2.0]);
        let err = c.validate(3, 1).unwrap_err();
        match err {
            GltfError::UnsupportedPath {
                animation,
                sampler,
                path,
                ..
            } => {
                assert_eq!(animation, 3);
                assert_eq!(sampler, 1);
                assert!(path.contains("descend"), "{path}");
            }
            other => panic!("wrong error: {other:?}"),
        }
    }

    #[test]
    fn a_non_finite_keyframe_time_is_rejected() {
        let c = Channel {
            node: 0,
            path: ChannelPath::Translation,
            keyframes: vec![key(f32::NAN, ChannelValue::Weight(0.0))],
        };
        assert!(c.validate(0, 0).is_err());
    }

    #[test]
    fn repeated_keyframe_times_are_allowed() {
        // A hold is expressed as two keyframes at one time, so this is valid
        // even though the span is zero.
        let c = translation_channel(0, &[1.0, 1.0], &[0.0, 1.0]);
        assert!(c.validate(0, 0).is_ok());
    }

    #[test]
    fn a_clip_runs_as_long_as_its_longest_channel() {
        let c = clip_of(vec![
            translation_channel(0, &[0.0, 1.0], &[0.0, 1.0]),
            translation_channel(1, &[0.0, 3.0], &[0.0, 1.0]),
        ]);
        assert!((c.duration() - 3.0).abs() < 1e-6);
    }

    #[test]
    fn a_clip_with_no_channels_is_zero_length() {
        assert_eq!(clip_of(Vec::new()).duration(), 0.0);
    }

    #[test]
    fn a_clip_with_no_channels_has_no_finite_start() {
        // Folding INFINITY over an empty iterator leaves it, which is why
        // Clip::wrap checks is_finite rather than trusting the start.
        assert!(clip_of(Vec::new()).start().is_infinite());
    }

    #[test]
    fn sampling_a_clip_returns_every_channel() {
        let c = clip_of(vec![
            translation_channel(0, &[0.0, 1.0], &[0.0, 1.0]),
            translation_channel(7, &[0.0, 1.0], &[0.0, 2.0]),
        ]);
        let sampled = c.sample(0.5);
        assert_eq!(sampled.len(), 2);
        assert_eq!(sampled[1].0, 7);
    }

    #[test]
    fn a_looping_clip_wraps_at_its_end() {
        let c = clip_of(vec![translation_channel(0, &[0.0, 1.0], &[0.0, 1.0])]);
        assert!((c.wrap(1.5, LoopMode::Loop) - 0.5).abs() < 1e-5);
    }

    #[test]
    fn a_once_clip_clamps_at_its_end() {
        let c = clip_of(vec![translation_channel(0, &[0.0, 1.0], &[0.0, 1.0])]);
        assert!((c.wrap(99.0, LoopMode::Once) - 1.0).abs() < 1e-5);
    }

    #[test]
    fn a_ping_pong_clip_reverses() {
        let c = clip_of(vec![translation_channel(0, &[0.0, 1.0], &[0.0, 1.0])]);
        // At 1.5s of a 1s clip, ping-pong is heading back down, so it is at
        // the same place as 0.5s.
        assert!((c.wrap(1.5, LoopMode::PingPong) - 0.5).abs() < 1e-5);
    }

    #[test]
    fn a_ping_pong_clip_stays_in_range() {
        let c = clip_of(vec![translation_channel(0, &[0.0, 1.0], &[0.0, 1.0])]);
        for t in 0..40 {
            let v = c.wrap(t as f32 * 0.37, LoopMode::PingPong);
            assert!((-1e-5..=1.0 + 1e-5).contains(&v), "t={t} v={v}");
        }
    }

    #[test]
    fn a_zero_length_clip_does_not_divide_by_zero() {
        let c = clip_of(Vec::new());
        for mode in [LoopMode::Once, LoopMode::Loop, LoopMode::PingPong] {
            let v = c.wrap(5.0, mode);
            assert!(v.is_finite(), "{mode:?} gave {v}");
        }
    }

    #[test]
    fn a_clip_starting_late_loops_from_its_own_start() {
        let c = clip_of(vec![translation_channel(0, &[2.0, 3.0], &[0.0, 1.0])]);
        // 3.5s is half a second past the end, so half a second into the clip.
        assert!(
            (c.wrap(3.5, LoopMode::Loop) - 0.5).abs() < 1e-5,
            "{:?}",
            c.wrap(3.5, LoopMode::Loop)
        );
    }

    #[test]
    fn a_playback_advances_its_clock() {
        let c = clip_of(vec![translation_channel(0, &[0.0, 1.0], &[0.0, 1.0])]);
        let mut p = Playback::new(0);
        p.advance(&c, 0.25);
        assert!((p.time - 0.25).abs() < 1e-5);
    }

    #[test]
    fn a_playback_wraps_and_keeps_running() {
        let c = clip_of(vec![translation_channel(0, &[0.0, 1.0], &[0.0, 1.0])]);
        let mut p = Playback::new(0);
        let at = p.advance(&c, 1.5);
        assert!((at - 0.5).abs() < 1e-5, "{at}");
        assert!(!p.is_finished(), "a looping clip never finishes");
    }

    #[test]
    fn a_once_playback_stops_at_the_end() {
        let c = clip_of(vec![translation_channel(0, &[0.0, 1.0], &[0.0, 1.0])]);
        let mut p = Playback::new(0);
        p.loop_mode = LoopMode::Once;
        let at = p.advance(&c, 5.0);
        assert!((at - 1.0).abs() < 1e-5);
        assert!(p.is_finished());
    }

    #[test]
    fn a_playback_runs_backwards_at_negative_speed() {
        let c = clip_of(vec![translation_channel(0, &[0.0, 1.0], &[0.0, 1.0])]);
        let mut p = Playback::new(0);
        p.speed = -1.0;
        p.time = 0.5;
        let at = p.advance(&c, 0.25);
        assert!((at - 0.25).abs() < 1e-5, "{at}");
    }

    #[test]
    fn seeking_resets_the_finished_flag() {
        let c = clip_of(vec![translation_channel(0, &[0.0, 1.0], &[0.0, 1.0])]);
        let mut p = Playback::new(0);
        p.loop_mode = LoopMode::Once;
        p.advance(&c, 5.0);
        assert!(p.is_finished());
        p.seek(0.0);
        assert!(!p.is_finished());
    }

    #[test]
    fn a_zero_length_clip_advances_to_a_finite_time() {
        let mut p = Playback::new(0);
        let at = p.advance(&clip_of(Vec::new()), 1.0);
        assert!(at.is_finite(), "{at}");
    }

    #[test]
    fn paths_round_trip_through_their_glTF_names() {
        for p in [
            ChannelPath::Translation,
            ChannelPath::Rotation,
            ChannelPath::Scale,
            ChannelPath::Weights,
        ] {
            assert_eq!(ChannelPath::parse(p.gltf_name()), Some(p));
        }
    }

    #[test]
    fn an_unknown_path_name_is_rejected() {
        assert_eq!(ChannelPath::parse("colour"), None);
    }

    #[test]
    fn composing_a_translation_offsets_the_base() {
        let base = glam::Affine3A::from_scale(Vec3::splat(2.0));
        let m = compose_local(
            ChannelPath::Translation,
            ChannelValue::Translation(Vec3::new(1.0, 0.0, 0.0)),
            base,
        );
        let p = m * Vec4::new(0.0, 0.0, 0.0, 1.0);
        assert!((p.x - 1.0).abs() < 1e-5, "{p:?}");
    }

    #[test]
    fn composing_a_scale_multiplies_the_base() {
        let base = glam::Affine3A::from_scale(Vec3::splat(2.0));
        let m = compose_local(
            ChannelPath::Scale,
            ChannelValue::Scale(Vec3::splat(3.0)),
            base,
        );
        let p = m * Vec4::new(1.0, 1.0, 1.0, 1.0);
        assert!((p.x - 6.0).abs() < 1e-4, "{p:?}");
    }

    #[test]
    fn composing_a_rotation_turns_the_base() {
        let base = glam::Affine3A::IDENTITY;
        let m = compose_local(
            ChannelPath::Rotation,
            ChannelValue::Rotation(Quat::from_rotation_z(std::f32::consts::FRAC_PI_2)),
            base,
        );
        let p = m * Vec4::new(1.0, 0.0, 0.0, 1.0);
        assert!((p.y - 1.0).abs() < 1e-5, "{p:?}");
    }

    #[test]
    fn composing_weights_leaves_the_base_alone() {
        let base = glam::Affine3A::from_scale(Vec3::splat(2.0));
        let m = compose_local(ChannelPath::Weights, ChannelValue::Weight(0.5), base);
        let p = m * Vec4::new(1.0, 1.0, 1.0, 1.0);
        assert!((p.x - 2.0).abs() < 1e-5, "{p:?}");
    }

    #[test]
    fn bone_matrices_match_the_skin_helper() {
        let skin = Skin {
            inverse_bind: vec![Mat4::IDENTITY],
            joints: vec![0],
        };
        let world = [Mat4::from_translation(Vec3::X)];
        assert_eq!(
            bone_matrices(&skin, &world),
            crate::mesh::skin_matrices(&skin, &world)
        );
    }
}
