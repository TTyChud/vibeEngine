//! Plays a short tone through the real audio device and reports what it did.
//!
//! Run with: cargo run -p vibe-audio --example tone

use std::time::Duration;

use vibe_audio::{Attenuation, AudioListener, LoopMode, Mixer, Vec3};

/// A 440 Hz tone, a quarter of a second.
fn tone(hz: f32, seconds: f32, rate: u32) -> Vec<f32> {
    let frames = (rate as f32 * seconds) as usize;
    (0..frames)
        .flat_map(|i| {
            let t = i as f32 / rate as f32;
            // A short fade at each end so the tone does not click.
            let env = (t / 0.01).min(1.0).min((seconds - t) / 0.05).max(0.0);
            let v = (t * hz * std::f32::consts::TAU).sin() * 0.3 * env;
            [v, v]
        })
        .collect()
}

fn main() {
    let mixer = match Mixer::open_default() {
        Ok(m) => m,
        Err(e) => {
            eprintln!("no audio device: {e}");
            return;
        }
    };
    println!("device: {}", mixer.device_name().unwrap_or("unknown"));
    println!("listening at default volume");

    let samples = tone(440.0, 0.5, 44_100);

    // Centre, unattenuated.
    let id = mixer
        .play(samples.clone(), LoopMode::Once)
        .expect("failed to start the tone");
    std::thread::sleep(Duration::from_millis(600));
    mixer.stop(id);

    // Same tone, placed to the right of the listener.
    let right = mixer
        .play(samples.clone(), LoopMode::Once)
        .expect("failed to start the tone");
    mixer.set_spatial(right, true);
    mixer.set_position(right, Vec3::new(20.0, 0.0, -5.0));
    std::thread::sleep(Duration::from_millis(600));
    mixer.stop(right);

    // And far away, which should be quieter.
    let far = mixer
        .play(samples.clone(), LoopMode::Once)
        .expect("failed to start");
    mixer.set_spatial(far, true);
    mixer.set_attenuation(far, Attenuation::new(1.0, 100.0));
    mixer.set_position(far, Vec3::new(0.0, 0.0, -80.0));
    std::thread::sleep(Duration::from_millis(600));
    mixer.stop(far);

    // A looping tone the listener walks past.
    let looped = mixer
        .play(tone(220.0, 0.2, 44_100), LoopMode::Loop)
        .expect("failed");
    for step in 0..10 {
        let x = -20.0 + step as f32 * 4.0;
        mixer.set_spatial(looped, true);
        mixer.set_position(looped, Vec3::new(x, 0.0, -5.0));
        std::thread::sleep(Duration::from_millis(60));
    }
    println!("voices still loaded: {}", mixer.voice_count());
    mixer.stop_all();
    std::thread::sleep(Duration::from_millis(100));

    // Mix a block by hand, which is what the audio callback does.
    let mut block = vec![0.0f32; 512];
    mixer
        .play(tone(440.0, 0.2, 44_100), LoopMode::Loop)
        .expect("failed");
    mixer.mix_block(&mut block);
    let peak = block.iter().fold(0.0f32, |a, b| a.max(b.abs()));
    println!("mixed block peak: {peak:.4} (non-zero means the voice is producing audio)");
    if peak == 0.0 {
        eprintln!("WARNING: a playing voice produced silence");
    }

    mixer.set_listener(AudioListener::at_origin());
    println!("done");
}
