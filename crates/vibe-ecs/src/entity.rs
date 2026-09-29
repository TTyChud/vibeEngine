//! Entity handles and UUIDs.

use std::fmt;

/// A lightweight handle to an entity: a slot index plus a generation.
///
/// The generation increments when a slot is recycled, so a handle to a
/// despawned entity never resolves to whatever now occupies that slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Entity {
    index: u32,
    generation: u32,
}

impl Entity {
    /// A handle for a slot that has never been used.
    pub const fn invalid() -> Entity {
        Entity {
            index: u32::MAX,
            generation: u32::MAX,
        }
    }

    pub(crate) const fn from_parts(index: u32, generation: u32) -> Entity {
        Entity { index, generation }
    }

    /// The slot index.
    pub const fn index(self) -> u32 {
        self.index
    }

    /// The generation, which increments on each slot reuse.
    pub const fn generation(self) -> u32 {
        self.generation
    }

    /// True when this handle was made by [`Entity::invalid`].
    pub const fn is_invalid(self) -> bool {
        self.index == u32::MAX
    }
}

impl fmt::Display for Entity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Entity({}v{})", self.index, self.generation)
    }
}

/// A stable 128-bit identifier that survives save/load.
///
/// Layout is a UUID v4, so values are interchangeable with other tools that
/// speak the same 16-byte form.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Default)]
pub struct Uuid([u8; 16]);

impl Uuid {
    /// The all-zero UUID, used as "no id".
    pub const NIL: Uuid = Uuid([0; 16]);

    /// A new random (version 4) UUID from thread-local entropy.
    pub fn new_v4() -> Uuid {
        thread_local! {
            static STATE: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
        }
        let bytes = STATE.with(|s| {
            let mut seed = s.get();
            if seed == 0 {
                seed = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .map(|d| d.as_nanos() as u64)
                    .unwrap_or(0x9E3779B97F4A7C15)
                    ^ (&s as *const _ as u64);
                if seed == 0 {
                    seed = 0x9E3779B97F4A7C15;
                }
            }
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            s.set(seed);
            let hi = seed ^ (seed >> 31);
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            s.set(seed);
            let lo = seed ^ (seed >> 29);
            (hi, lo)
        });
        let (hi, lo) = bytes;
        let mut b = [0u8; 16];
        b[..8].copy_from_slice(&hi.to_le_bytes());
        b[8..].copy_from_slice(&lo.to_le_bytes());
        b[6] = (b[6] & 0x0f) | 0x40;
        b[8] = (b[8] & 0x3f) | 0x80;
        Uuid(b)
    }

    /// Build a UUID from raw bytes.
    pub const fn from_bytes(bytes: [u8; 16]) -> Uuid {
        Uuid(bytes)
    }

    /// The raw bytes.
    pub const fn as_bytes(&self) -> &[u8; 16] {
        &self.0
    }

    /// True when this is the all-zero UUID.
    pub fn is_nil(&self) -> bool {
        self.0 == [0u8; 16]
    }

    /// Parse the hyphenated form, e.g. `9f1c2d3e-4a5b-4c6d-8e9f-0a1b2c3d4e5f`.
    pub fn parse(s: &str) -> Option<Uuid> {
        let hex: Vec<u8> = s.bytes().filter(|b| *b != b'-').collect();
        if hex.len() != 32 {
            return None;
        }
        let mut out = [0u8; 16];
        for i in 0..16 {
            let hi = (hex[i * 2] as char).to_digit(16)?;
            let lo = (hex[i * 2 + 1] as char).to_digit(16)?;
            out[i] = ((hi << 4) | lo) as u8;
        }
        Some(Uuid(out))
    }
}

impl fmt::Display for Uuid {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (i, b) in self.0.iter().enumerate() {
            if matches!(i, 4 | 6 | 8 | 10) {
                write!(f, "-")?;
            }
            write!(f, "{b:02x}")?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn invalid_entity_is_recognised() {
        assert!(Entity::invalid().is_invalid());
        assert!(!Entity::from_parts(0, 0).is_invalid());
    }

    #[test]
    fn entity_parts_round_trip() {
        let e = Entity::from_parts(7, 3);
        assert_eq!(e.index(), 7);
        assert_eq!(e.generation(), 3);
    }

    #[test]
    fn entity_display_includes_generation() {
        let s = Entity::from_parts(2, 5).to_string();
        assert!(s.contains("2"));
        assert!(s.contains("5"));
    }

    #[test]
    fn entities_with_different_generation_are_not_equal() {
        assert_ne!(Entity::from_parts(1, 1), Entity::from_parts(1, 2));
    }

    #[test]
    fn nil_uuid_is_nil() {
        assert!(Uuid::NIL.is_nil());
        assert!(!Uuid::new_v4().is_nil());
    }

    #[test]
    fn uuid_v4_has_version_and_variant_nibbles() {
        let u = Uuid::new_v4();
        assert_eq!(u.0[6] >> 4, 4, "version nibble");
        assert_eq!(u.0[8] >> 6, 0b10, "variant bits");
    }

    #[test]
    fn uuid_v4_is_usually_unique() {
        let set: HashSet<_> = (0..2000).map(|_| Uuid::new_v4()).collect();
        assert!(
            set.len() > 1990,
            "expected few collisions, got {} unique",
            set.len()
        );
    }

    #[test]
    fn uuid_display_is_hyphenated_36_chars() {
        let s = Uuid::new_v4().to_string();
        assert_eq!(s.len(), 36);
        assert_eq!(s.chars().filter(|c| *c == '-').count(), 4);
    }

    #[test]
    fn uuid_round_trips_through_string() {
        let u = Uuid::new_v4();
        assert_eq!(Uuid::parse(&u.to_string()), Some(u));
    }

    #[test]
    fn uuid_parse_rejects_bad_input() {
        assert_eq!(Uuid::parse(""), None);
        assert_eq!(Uuid::parse("not-a-uuid"), None);
        assert_eq!(Uuid::parse("zzzzzzzz-zzzz-zzzz-zzzz-zzzzzzzzzzzz"), None);
    }

    #[test]
    fn uuid_from_bytes_round_trips() {
        let bytes: [u8; 16] = core::array::from_fn(|i| i as u8);
        assert_eq!(Uuid::from_bytes(bytes).as_bytes(), &bytes);
    }

    #[test]
    fn uuid_from_multiple_threads_does_not_collide() {
        let handles: Vec<_> = (0..4)
            .map(|_| std::thread::spawn(|| (0..500).map(|_| Uuid::new_v4()).collect::<Vec<_>>()))
            .collect();
        let all: HashSet<Uuid> = handles
            .into_iter()
            .flat_map(|h| h.join().unwrap())
            .collect();
        assert!(all.len() > 1900, "got {} unique uuids", all.len());
    }
}
