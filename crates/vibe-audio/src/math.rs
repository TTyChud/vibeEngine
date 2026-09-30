//! Spatialisation maths: attenuation, panning and doppler.
//!
//! Kept free of cpal so the placement rules can be tested exactly, with no
//! sound card. The rules match the inverse-distance model most engines use:
//! gain falls off between a reference distance and a rolloff distance, and
//! nothing is ever fully inaudible until it passes the rolloff point.

// A small vector of our own, so callers do not need glam for audio placement.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Vec3 {
    /// East-west, right positive.
    pub x: f32,
    /// Up-down, up positive.
    pub y: f32,
    /// Forward-back, forward negative to match a right-handed camera looking
    /// down -Z.
    pub z: f32,
}

impl Vec3 {
    /// The origin.
    pub const ZERO: Vec3 = Vec3 {
        x: 0.0,
        y: 0.0,
        z: 0.0,
    };

    /// A vector from three components.
    pub const fn new(x: f32, y: f32, z: f32) -> Vec3 {
        Vec3 { x, y, z }
    }

    /// Distance to another point.
    pub fn distance_to(self, other: Vec3) -> f32 {
        let d = self.sub(other);
        (d.x * d.x + d.y * d.y + d.z * d.z).sqrt()
    }

    /// Component-wise difference.
    pub fn sub(self, other: Vec3) -> Vec3 {
        Vec3::new(self.x - other.x, self.y - other.y, self.z - other.z)
    }
}

impl From<glam::Vec3> for Vec3 {
    fn from(v: glam::Vec3) -> Vec3 {
        Vec3::new(v.x, v.y, v.z)
    }
}

impl From<Vec3> for glam::Vec3 {
    fn from(v: Vec3) -> glam::Vec3 {
        glam::Vec3::new(v.x, v.y, v.z)
    }
}

/// How a source's loudness falls with distance.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Attenuation {
    /// Distance at which gain is 1.0, so a source at the listener is not
    /// boosted by the falloff curve.
    pub reference_distance: f32,
    /// Distance at which the source becomes inaudible.
    pub rolloff_distance: f32,
}

impl Default for Attenuation {
    fn default() -> Self {
        Attenuation {
            reference_distance: 1.0,
            rolloff_distance: 100.0,
        }
    }
}

impl Attenuation {
    /// A model with the given reference and rolloff distances.
    pub fn new(reference_distance: f32, rolloff_distance: f32) -> Attenuation {
        Attenuation {
            reference_distance,
            rolloff_distance,
        }
    }

    /// True when the distances are usable.
    ///
    /// A rolloff at or below the reference would make the curve either flat or
    /// inverted, so a zero or negative span is rejected rather than clamped.
    pub fn is_valid(&self) -> bool {
        self.reference_distance > 0.0
            && self.rolloff_distance > self.reference_distance
            && self.rolloff_distance.is_finite()
    }
}

/// Gain for a source `distance` from the listener, in `0.0..=1.0`.
///
/// Inverse-distance inside the rolloff span, and zero beyond it. Linear inside
/// the reference distance so a source at the listener plays at full volume
/// rather than being boosted by the curve.
pub fn distance_gain(distance: f32, attenuation: Attenuation) -> f32 {
    if !distance.is_finite() || distance < 0.0 {
        return 0.0;
    }
    let d = distance.clamp(0.0, attenuation.rolloff_distance);
    if d <= attenuation.reference_distance {
        // Full volume up to the reference distance, rising linearly into it.
        (d / attenuation.reference_distance)
            .clamp(0.0, 1.0)
            .max(if d == 0.0 { 1.0 } else { 0.0 })
    } else {
        let span = attenuation.rolloff_distance - attenuation.reference_distance;
        if span <= 0.0 {
            return 0.0;
        }
        1.0 - (d - attenuation.reference_distance) / span
    }
}

/// Clamp a distance into the audible range.
pub fn clamp_distance(distance: f32, rolloff: f32) -> f32 {
    if distance.is_finite() {
        distance.clamp(0.0, rolloff.max(0.0))
    } else {
        rolloff.max(0.0)
    }
}

/// Stereo pan for a source offset along X relative to the listener.
///
/// Returns `-1.0` hard left, `0.0` centre, `1.0` hard right. A source
/// directly at or behind the listener is centred, because panning is an
/// azimuth effect and there is no azimuth to speak of at zero offset.
pub fn pan_for_x(offset_x: f32) -> f32 {
    if !offset_x.is_finite() {
        return 0.0;
    }
    offset_x.clamp(-1.0, 1.0)
}

/// Playback rate multiplier for a source `distance` away, for doppler.
///
/// A source moving away plays slower. Capped, because an uncapped doppler shift
/// on a fast-moving listener sounds like a glitch rather than an effect.
pub fn pitch_for_distance(distance: f32, max_shift: f32) -> f32 {
    if !distance.is_finite() || !max_shift.is_finite() || max_shift <= 0.0 {
        return 1.0;
    }
    let shift = (distance * 0.01).clamp(-max_shift, max_shift);
    (1.0 - shift).clamp(1.0 - max_shift, 1.0 + max_shift)
}

/// Where the listener is and which way it faces.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AudioListener {
    /// The listener's position.
    pub position: Vec3,
    /// The direction the listener faces, in the listener's local space.
    ///
    /// The forward component is expected negative for a camera looking down -Z;
    /// only the horizontal forward component is used for panning.
    pub forward: glam::Vec3,
    /// The listener's up axis, needed to build the right vector.
    pub up: glam::Vec3,
    /// Master volume in `0.0..=1.0`.
    pub volume: f32,
}

impl Default for AudioListener {
    fn default() -> Self {
        AudioListener {
            position: Vec3::ZERO,
            forward: glam::Vec3::NEG_Z,
            up: glam::Vec3::Y,
            volume: 1.0,
        }
    }
}

impl AudioListener {
    /// A listener at the origin facing -Z.
    pub fn at_origin() -> AudioListener {
        AudioListener::default()
    }

    /// The listener's right vector, derived from forward and up.
    pub fn right(&self) -> glam::Vec3 {
        self.forward.cross(self.up).normalize_or_zero()
    }

    /// The listener's forward direction flattened onto the horizontal plane.
    ///
    /// Panning is a horizontal effect, so the vertical component is dropped
    /// rather than skewing the azimuth.
    pub fn forward_flat(&self) -> glam::Vec3 {
        let mut f = self.forward;
        f.y = 0.0;
        f.normalize_or_zero()
    }

    /// Pan and gain for a source at `position`.
    ///
    /// Returns `(pan, gain)`, with pan in `-1.0..=1.0` and gain in
    /// `0.0..=1.0`. The source's offset is projected onto the listener's right
    /// axis, so turning the listener changes what is heard on the left.
    pub fn spatialise(&self, position: Vec3, attenuation: Attenuation) -> (f32, f32) {
        let offset = glam::Vec3::from(position.sub(self.position));
        let distance = offset.length();
        let gain = distance_gain(distance, attenuation) * self.volume;

        let right = self.right();
        let forward = self.forward_flat();
        // Offset in the listener's frame: +x is right, +z is behind.
        let local_right = offset.dot(right);
        let local_forward = offset.dot(forward);
        // A source in front is positive distance, behind negative; using the
        // forward component keeps a source behind the listener panned correctly
        // rather than flipped.
        let azimuth = if local_forward.abs() + local_right.abs() < f32::EPSILON {
            0.0
        } else {
            (local_right / (local_right.abs() + local_forward.abs())).clamp(-1.0, 1.0)
        };
        (pan_for_x(azimuth), gain.clamp(0.0, 1.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn distance_is_euclidean() {
        let a = Vec3::new(0.0, 0.0, 0.0);
        let b = Vec3::new(3.0, 4.0, 0.0);
        assert!((a.distance_to(b) - 5.0).abs() < 1e-5);
    }

    #[test]
    fn a_vec3_converts_both_ways() {
        let g = glam::Vec3::new(1.0, 2.0, 3.0);
        let a = Vec3::from(g);
        assert_eq!(a, Vec3::new(1.0, 2.0, 3.0));
        assert_eq!(glam::Vec3::from(a), g);
    }

    #[test]
    fn a_source_at_the_listener_is_full_volume() {
        let a = Attenuation::default();
        assert!((distance_gain(0.0, a) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn gain_falls_with_distance() {
        let a = Attenuation::new(1.0, 100.0);
        let near = distance_gain(10.0, a);
        let far = distance_gain(50.0, a);
        assert!(
            near > far,
            "a nearer source must be louder: {near} vs {far}"
        );
    }

    #[test]
    fn gain_is_zero_at_the_rolloff_distance() {
        let a = Attenuation::new(1.0, 100.0);
        assert!(distance_gain(100.0, a).abs() < 1e-6);
        assert_eq!(distance_gain(500.0, a), 0.0, "beyond rolloff is silent");
    }

    #[test]
    fn gain_never_exceeds_one() {
        let a = Attenuation::new(1.0, 100.0);
        for d in [0.0, 0.5, 1.0, 10.0, 99.0] {
            let g = distance_gain(d, a);
            assert!((0.0..=1.0).contains(&g), "gain {g} out of range at {d}");
        }
    }

    #[test]
    fn a_non_finite_distance_is_silent_not_a_panic() {
        let a = Attenuation::default();
        assert_eq!(distance_gain(f32::NAN, a), 0.0);
        assert_eq!(distance_gain(f32::INFINITY, a), 0.0);
        assert_eq!(distance_gain(-1.0, a), 0.0);
    }

    #[test]
    fn an_invalid_attenuation_is_reported() {
        assert!(Attenuation::new(1.0, 100.0).is_valid());
        assert!(!Attenuation::new(0.0, 100.0).is_valid(), "zero reference");
        assert!(
            !Attenuation::new(10.0, 10.0).is_valid(),
            "rolloff at reference"
        );
        assert!(!Attenuation::new(10.0, 5.0).is_valid(), "inverted span");
    }

    #[test]
    fn an_invalid_attenuation_gives_silence_rather_than_a_huge_gain() {
        let bad = Attenuation::new(10.0, 10.0);
        assert!(distance_gain(20.0, bad) >= 0.0);
        assert!(
            distance_gain(20.0, bad) <= 1.0,
            "must not exceed full volume"
        );
    }

    #[test]
    fn clamping_a_distance_stays_in_range() {
        assert_eq!(clamp_distance(-5.0, 100.0), 0.0);
        assert_eq!(clamp_distance(500.0, 100.0), 100.0);
        assert_eq!(clamp_distance(50.0, 100.0), 50.0);
    }

    #[test]
    fn clamping_a_nan_distance_falls_to_the_rolloff() {
        assert_eq!(clamp_distance(f32::NAN, 100.0), 100.0);
    }

    #[test]
    fn panning_maps_left_and_right() {
        assert_eq!(pan_for_x(-1.0), -1.0);
        assert_eq!(pan_for_x(0.0), 0.0);
        assert_eq!(pan_for_x(1.0), 1.0);
    }

    #[test]
    fn panning_clamps_and_survives_nan() {
        assert_eq!(pan_for_x(5.0), 1.0);
        assert_eq!(pan_for_x(-5.0), -1.0);
        assert_eq!(pan_for_x(f32::NAN), 0.0);
    }

    #[test]
    fn a_source_to_the_right_pans_right() {
        let l = AudioListener::at_origin();
        let (pan, _) = l.spatialise(Vec3::new(5.0, 0.0, -5.0), Attenuation::default());
        assert!(pan > 0.0, "a source to the right pans right, got {pan}");
    }

    #[test]
    fn a_source_to_the_left_pans_left() {
        let l = AudioListener::at_origin();
        let (pan, _) = l.spatialise(Vec3::new(-5.0, 0.0, -5.0), Attenuation::default());
        assert!(pan < 0.0, "a source to the left pans left, got {pan}");
    }

    #[test]
    fn a_source_directly_ahead_is_centred() {
        let l = AudioListener::at_origin();
        let (pan, _) = l.spatialise(Vec3::new(0.0, 0.0, -10.0), Attenuation::default());
        assert!(
            pan.abs() < 1e-4,
            "a source straight ahead is centred, got {pan}"
        );
    }

    #[test]
    fn a_source_directly_behind_is_centred() {
        // There is no azimuth directly behind, so panning must not flip.
        let l = AudioListener::at_origin();
        let (pan, _) = l.spatialise(Vec3::new(0.0, 0.0, 10.0), Attenuation::default());
        assert!(pan.abs() < 1e-4, "a source behind is centred, got {pan}");
    }

    #[test]
    fn turning_the_listener_changes_what_is_heard_on_the_left() {
        let source = Vec3::new(5.0, 0.0, -5.0);
        let facing_z = AudioListener::at_origin();
        let facing_x = AudioListener {
            forward: glam::Vec3::X,
            ..AudioListener::at_origin()
        };
        let (a, _) = facing_z.spatialise(source, Attenuation::default());
        let (b, _) = facing_x.spatialise(source, Attenuation::default());
        assert!(
            a > 0.0 && b < 0.0,
            "turning the listener should flip the pan: {a} vs {b}"
        );
    }

    #[test]
    fn the_listener_offsets_every_source() {
        let at_origin = AudioListener::at_origin();
        let moved = AudioListener {
            position: Vec3::new(-20.0, 0.0, 0.0),
            ..AudioListener::at_origin()
        };
        let source = Vec3::new(0.0, 0.0, 0.0);
        let (pan_a, gain_a) = at_origin.spatialise(source, Attenuation::new(1.0, 100.0));
        let (pan_b, gain_b) = moved.spatialise(source, Attenuation::new(1.0, 100.0));
        assert!(gain_a > gain_b, "moving toward a source makes it louder");
        assert!(
            pan_a < pan_b,
            "moving changes its position: {pan_a} vs {pan_b}"
        );
    }

    #[test]
    fn listener_volume_scales_the_gain() {
        let source = Vec3::new(0.0, 0.0, -10.0);
        let full = AudioListener::at_origin();
        let half = AudioListener {
            volume: 0.5,
            ..full
        };
        let (_, g_full) = full.spatialise(source, Attenuation::new(1.0, 100.0));
        let (_, g_half) = half.spatialise(source, Attenuation::new(1.0, 100.0));
        assert!((g_half - g_full * 0.5).abs() < 1e-5);
    }

    #[test]
    fn the_listener_right_is_perpendicular_to_forward() {
        let l = AudioListener::at_origin();
        let r = l.right();
        assert!(
            l.forward.dot(r).abs() < 1e-5,
            "right must be perpendicular to forward"
        );
    }

    #[test]
    fn the_flattened_forward_drops_the_vertical() {
        let l = AudioListener {
            forward: glam::Vec3::new(0.0, 5.0, -1.0),
            ..AudioListener::at_origin()
        };
        assert!(
            l.forward_flat().y.abs() < 1e-6,
            "panning has no vertical component"
        );
    }

    #[test]
    fn a_source_at_the_listener_does_not_panic() {
        let l = AudioListener::at_origin();
        let (pan, gain) = l.spatialise(Vec3::ZERO, Attenuation::default());
        assert!(pan.abs() < 1e-6);
        assert!((gain - 1.0).abs() < 1e-6);
    }

    #[test]
    fn doppler_shifts_with_distance_but_is_capped() {
        let near = pitch_for_distance(0.0, 0.1);
        let far = pitch_for_distance(100.0, 0.1);
        assert!((near - 1.0).abs() < 1e-6, "no shift at zero distance");
        assert!(far < near, "a distant source plays slower");
        assert!(
            pitch_for_distance(1e9, 0.1) >= 1.0 - 0.1 - 1e-6,
            "shift is capped"
        );
    }

    #[test]
    fn doppler_degenerates_to_no_shift() {
        assert_eq!(
            pitch_for_distance(50.0, 0.0),
            1.0,
            "zero cap means no shift"
        );
        assert_eq!(pitch_for_distance(f32::NAN, 0.1), 1.0);
    }
}
