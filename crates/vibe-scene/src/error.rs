//! Errors from scene loading and saving.

/// A scene could not be read or written.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum SceneError {
    /// The file was not valid YAML, or did not match the scene schema.
    #[error("could not parse scene: {0}")]
    Parse(String),
    /// The file declares a version this build cannot read.
    #[error("scene version {found} is newer than the supported version {supported}")]
    UnsupportedVersion {
        /// The version in the file.
        found: u32,
        /// The highest version this build reads.
        supported: u32,
    },
    /// An entity's UUID was not a valid UUID string.
    #[error("entity has invalid uuid {uuid:?}")]
    BadUuid {
        /// The offending string.
        uuid: String,
    },
    /// The scene had two entities with the same UUID.
    #[error("duplicate entity uuid {0}")]
    DuplicateUuid(String),
    /// A component value did not match the component it was filed under.
    #[error("component {component} on entity {entity} has the wrong value type")]
    MismatchedValue {
        /// The component name.
        component: String,
        /// The entity's UUID.
        entity: String,
    },
}
