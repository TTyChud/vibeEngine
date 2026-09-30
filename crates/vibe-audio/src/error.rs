//! Errors from audio setup and playback.

/// An audio operation could not be completed.
#[derive(Debug, thiserror::Error)]
pub enum AudioError {
    /// No output device is available on this machine.
    #[error("no audio output device")]
    NoOutputDevice,
    /// The device rejected the requested stream configuration.
    #[error("unsupported stream configuration: {0}")]
    UnsupportedConfig(String),
    /// The device could not be opened.
    #[error("could not open the audio device: {0}")]
    OpenDevice(String),
    /// A voice was referenced that does not exist.
    #[error("no voice with id {0}")]
    NoSuchVoice(u64),
    /// A voice was given samples, looping, before it was started.
    #[error("voice {0} is not started")]
    NotStarted(u64),
}
