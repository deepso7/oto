use super::*;
use crate::signal::{db, noisy_speech, pink_noise, rms, sine, speech_like, white_noise};

const SR: usize = SAMPLE_RATE as usize;

#[test]
fn odd_block_sizes_come_out_delayed_by_exactly_one_frame() {
    let mut a = FrameAdapter::new(Passthrough::new(480));
    let input: Vec<f32> = (0..5000).map(|i| i as f32).collect();
    let mut output = vec![0.0; input.len()];
    let mut at = 0;
    for block in [1, 7, 128, 479, 480, 481, 1024, 2300].iter().cycle() {
        if at == input.len() {
            break;
        }
        let end = (at + block).min(input.len());
        a.process(&input[at..end], &mut output[at..end]);
        at = end;
    }
    assert_eq!(a.latency(), 480);
    assert!(output[..480].iter().all(|&s| s == 0.0));
    assert_eq!(&output[480..], &input[..input.len() - 480]);
}

#[test]
fn reset_clears_the_tail() {
    let mut a = FrameAdapter::new(Passthrough::new(4));
    let mut out = [0.0; 6];
    a.process(&[1.0; 6], &mut out);
    a.reset();
    a.process(&[0.0; 6], &mut out);
    assert_eq!(out, [0.0; 6]);
}

#[test]
fn in_place_matches_out_of_place() {
    let input: Vec<f32> = (0..3000).map(|i| (i as f32 * 0.37).sin()).collect();
    let mut a = FrameAdapter::new(Passthrough::new(480));
    let mut b = FrameAdapter::new(Passthrough::new(480));
    let mut out = vec![0.0; input.len()];
    a.process(&input, &mut out);
    let mut buf = input.clone();
    b.process_in_place(&mut buf);
    assert_eq!(out, buf);
}

/// Steady-state gain of the high-pass at `freq`, in dB.
fn hp_gain_db(freq: f32) -> f32 {
    let mut hp = HighPass::new(SR as f64, HIGH_PASS_HZ as f64);
    let mut x = sine(freq, 0.5, SR * 2);
    let reference = rms(&x[SR..]);
    hp.process(&mut x);
    db(rms(&x[SR..]) / reference)
}

#[test]
fn high_pass_kills_20_hz_and_passes_1_khz() {
    let g20 = hp_gain_db(20.0);
    let g80 = hp_gain_db(80.0);
    let g1k = hp_gain_db(1000.0);
    assert!(g20 < -40.0, "20 Hz: {g20:.1} dB");
    assert!((g80 + 3.0).abs() < 0.3, "80 Hz corner: {g80:.2} dB");
    assert!(g1k.abs() < 0.05, "1 kHz: {g1k:.3} dB");
}

#[test]
fn high_pass_removes_dc_and_settles_to_exact_zero() {
    let mut hp = HighPass::new(SR as f64, 80.0);
    let mut x = vec![0.3; SR];
    hp.process(&mut x);
    assert!(x[SR - 1].abs() < 1e-6);
    let mut z = vec![0.0; SR];
    hp.process(&mut z);
    assert_eq!(
        z[SR - 1],
        0.0,
        "state must flush so digital silence stays digital silence"
    );
}

#[test]
fn limiter_never_exceeds_ceiling_on_hot_signal() {
    let mut lim = Limiter::new(SR as f32, -1.0, 50.0);
    let ceiling = lim.ceiling();
    // +12 dBFS sine plus noise bursts and a few non-finite samples.
    let mut x: Vec<f32> = sine(440.0, 4.0, SR)
        .iter()
        .zip(white_noise(3.0, SR, 9))
        .map(|(s, n)| s + n)
        .collect();
    x[100] = f32::NAN;
    x[200] = f32::INFINITY;
    lim.process(&mut x);
    let peak = x.iter().fold(0.0f32, |m, s| m.max(s.abs()));
    assert!(x.iter().all(|s| s.is_finite()));
    assert!(peak <= ceiling, "peak {peak} over ceiling {ceiling}");
    assert!(
        peak > ceiling * 0.99,
        "hot signal should be pushed to the ceiling"
    );
}

#[test]
fn limiter_leaves_quiet_signal_alone() {
    let mut lim = Limiter::new(SR as f32, -1.0, 50.0);
    let x = sine(1000.0, 0.5, SR / 10);
    let mut y = x.clone();
    lim.process(&mut y);
    assert_eq!(x, y);
}

#[test]
fn chain_output_never_exceeds_ceiling() {
    let ceiling = 10f32.powf(LIMITER_CEILING_DB / 20.0);
    for enabled in [true, false] {
        let mut chain = Chain::new();
        chain.set_enabled(enabled);
        let mut x: Vec<f32> = speech_like(SR * 2, 3).iter().map(|s| s * 8.0).collect();
        for block in x.chunks_mut(256) {
            chain.process(block);
        }
        let peak = x.iter().fold(0.0f32, |m, s| m.max(s.abs()));
        assert!(peak <= ceiling, "enabled={enabled}: peak {peak}");
    }
}

/// `x` through the chain's high-pass alone: the reference for "how much did denoising remove".
fn high_passed(x: &[f32]) -> Vec<f32> {
    let mut y = x.to_vec();
    HighPass::new(SR as f64, HIGH_PASS_HZ as f64).process(&mut y);
    y
}

fn run_chain(chain: &mut Chain, input: &[f32], block: usize) -> Vec<f32> {
    let mut x = input.to_vec();
    for b in x.chunks_mut(block) {
        chain.process(b);
    }
    x
}

#[test]
fn rnnoise_chain_removes_room_noise() {
    // A -50 dBFS pink noise floor, typical of a laptop mic in a quiet room. RNNoise on noise
    // *alone* is uneven (its VAD sometimes latches onto it), so judge it over several takes.
    let reductions: Vec<f32> = [3, 11, 12, 99]
        .into_iter()
        .map(|seed| {
            let mut chain = Chain::with_model(Model::RNNoise).unwrap();
            chain.set_max_attenuation_db(f32::INFINITY);
            let noise = pink_noise(0.003, SR * 4, seed);
            let out = run_chain(&mut chain, &noise, 480);
            // Skip two seconds while the model converges.
            db(rms(&out[2 * SR..]) / rms(&high_passed(&noise)[2 * SR..]))
        })
        .collect();
    let mean = reductions.iter().sum::<f32>() / reductions.len() as f32;
    assert!(mean < -20.0, "pink noise reduced by {reductions:?} dB");
    assert!(reductions.iter().all(|&r| r < -5.0), "{reductions:?}");
}

/// RNNoise is erratic on flat full-band white noise *alone* (anywhere from 1 to 40 dB of
/// reduction depending on level: it looks like a sustained fricative). With speech present it
/// tracks it properly, which is the case that matters.
#[test]
fn rnnoise_chain_removes_white_noise_and_keeps_speech_like_signal() {
    let mut chain = Chain::with_model(Model::RNNoise).unwrap();
    chain.set_max_attenuation_db(f32::INFINITY);
    let speech = speech_like(SR * 4, 5);
    let noise = white_noise(0.05, SR * 4, 6);
    let noisy: Vec<f32> = speech.iter().zip(&noise).map(|(s, n)| s + n).collect();

    let clean_out = run_chain(&mut chain, &speech, 480);
    chain.reset();
    let noisy_out = run_chain(&mut chain, &noisy, 480);

    let lat = chain.latency();
    let kept = db(rms(&clean_out[SR + lat..]) / rms(&speech[SR..speech.len() - lat]));
    assert!(kept > -4.0, "speech-like signal lost {kept:.1} dB");

    // SNR improvement: residual against the (identically processed) clean signal.
    let in_snr = db(rms(&speech[SR..]) / rms(&noise[SR..]));
    let resid: Vec<f32> = noisy_out[SR..]
        .iter()
        .zip(&clean_out[SR..])
        .map(|(a, b)| a - b)
        .collect();
    let out_snr = db(rms(&clean_out[SR..]) / rms(&resid));
    assert!(
        out_snr > in_snr + 8.0,
        "SNR {in_snr:.1} dB -> {out_snr:.1} dB"
    );
}

#[test]
fn strength_limits_attenuation() {
    let noise = pink_noise(0.003, SR * 4, 12);
    let reference = high_passed(&noise);
    let mut reductions = Vec::new();
    for limit in [12.0, 24.0, f32::INFINITY] {
        let mut chain = Chain::new();
        chain.set_max_attenuation_db(limit);
        let out = run_chain(&mut chain, &noise, 480);
        let r = -db(rms(&out[2 * SR..]) / rms(&reference[2 * SR..]));
        assert!(r <= limit + 0.5, "limit {limit} dB, attenuated {r:.1} dB");
        reductions.push(r);
    }
    assert!(
        reductions[0] > 11.0,
        "Light should still reduce: {reductions:?}"
    );
    assert!(
        reductions[0] < reductions[1] && reductions[1] < reductions[2],
        "{reductions:?}"
    );
}

#[test]
fn disabled_chain_is_transparent_above_the_high_pass_and_runs_no_inference() {
    for model in Model::ALL.into_iter().filter(|m| m.available()) {
        let mut chain = Chain::with_model(model).unwrap();
        chain.set_enabled(false);
        let before = chain.inference_frames();
        for freq in [200.0, 1000.0, 5000.0, 15000.0] {
            let x = sine(freq, 0.25, SR * 2);
            let y = run_chain(&mut chain, &x, 333);
            let g = db(rms(&y[SR..]) / rms(&x[SR..]));
            assert!(g.abs() < 0.1, "{model:?} {freq} Hz: {g:.3} dB");
        }
        assert_eq!(
            chain.inference_frames(),
            before,
            "{model:?} ran inference while disabled"
        );
    }
}

#[test]
fn disabled_chain_is_a_pure_delay_after_the_high_pass() {
    let mut chain = Chain::new();
    chain.set_enabled(false);
    let x = noisy_speech(SR, 4);
    let y = run_chain(&mut chain, &x, 100);
    let mut hp = HighPass::new(SR as f64, HIGH_PASS_HZ as f64);
    let mut want = x.clone();
    hp.process(&mut want);
    let lat = chain.latency();
    assert!(y[..lat].iter().all(|&s| s == 0.0));
    for (a, b) in y[lat..].iter().zip(&want) {
        assert!((a - b).abs() < 1e-6);
    }
}

/// Delay of `y` relative to `x` by brute-force cross-correlation over `0..max_lag`.
fn measured_delay(x: &[f32], y: &[f32], max_lag: usize) -> usize {
    let span = SR / 2..x.len() - max_lag;
    (0..max_lag)
        .max_by(|&a, &b| {
            let c = |lag: usize| {
                span.clone()
                    .map(|i| x[i] as f64 * y[i + lag] as f64)
                    .sum::<f64>()
            };
            c(a).total_cmp(&c(b))
        })
        .unwrap()
}

#[test]
fn reported_latency_matches_measured_delay() {
    for model in Model::ALL.into_iter().filter(|m| m.available()) {
        let mut chain = Chain::with_model(model).unwrap();
        let x = speech_like(SR * 2, 21);
        let y = run_chain(&mut chain, &x, 480);
        let d = measured_delay(&x, &y, 4000);
        let lat = chain.latency();
        assert!(
            d.abs_diff(lat) <= 2,
            "{model:?}: reported {lat}, measured {d}"
        );
        chain.set_enabled(false);
    }
}

#[test]
fn latency_is_the_same_enabled_or_disabled() {
    let mut chain = Chain::new();
    let on = chain.latency();
    chain.set_enabled(false);
    assert_eq!(chain.latency(), on);
    assert_eq!(on, 960, "RNNoise: 10 ms buffering + 10 ms window overlap");
}

#[test]
fn block_size_does_not_change_output() {
    let x = noisy_speech(SR * 2, 8);
    let mut big = Chain::new();
    let want = run_chain(&mut big, &x, x.len());

    let mut small = Chain::new();
    let mut y = x.clone();
    let mut at = 0;
    for block in [1, 7, 64, 128, 333, 479, 480, 481, 1024, 4096]
        .iter()
        .cycle()
    {
        if at == y.len() {
            break;
        }
        let end = (at + block).min(y.len());
        small.process(&mut y[at..end]);
        at = end;
    }
    assert_eq!(want, y);
}

#[test]
fn digital_silence_skips_inference_and_stays_silent() {
    let mut chain = Chain::new();
    let speech = noisy_speech(SR / 2, 2);
    run_chain(&mut chain, &speech, 480);
    let after_speech = chain.inference_frames();
    // A second of zeros: a few frames to flush the tails, then nothing.
    let out = run_chain(&mut chain, &vec![0.0; SR], 480);
    let used = chain.inference_frames() - after_speech;
    assert!(used < 30, "ran {used} frames on silence");
    assert!(out[SR / 2..].iter().all(|&s| s == 0.0));
    // And it comes back.
    run_chain(&mut chain, &speech, 480);
    assert!(chain.inference_frames() > after_speech + used + 40);
}

#[test]
fn toggling_is_click_free() {
    let mut chain = Chain::new();
    chain.set_max_attenuation_db(f32::INFINITY);
    let x = sine(440.0, 0.3, SR * 2);
    let mut y = x.clone();
    let mut max_jump = 0.0f32;
    let mut prev = 0.0f32;
    for (i, block) in y.chunks_mut(240).enumerate() {
        if i % 20 == 10 {
            chain.set_enabled(!chain.enabled());
        }
        chain.process(block);
        for &s in block.iter() {
            if i > 100 {
                max_jump = max_jump.max((s - prev).abs());
            }
            prev = s;
        }
    }
    // A 440 Hz sine at 0.3 moves at most 0.3 * 2π * 440 / 48000 ≈ 0.017 per sample.
    assert!(max_jump < 0.05, "max sample-to-sample jump {max_jump}");
}

#[test]
fn unavailable_model_is_an_error_not_a_panic() {
    if !Model::DeepFilterNet3.available() {
        assert!(Chain::with_model(Model::DeepFilterNet3).is_err());
        assert!(try_bench(Model::DeepFilterNet3, 0.1).is_err());
    }
}

#[test]
fn bench_reports_sane_numbers() {
    let r = bench(Model::RNNoise, 0.5);
    assert_eq!(r.frames, 50);
    assert!(r.p50_us > 0.0 && r.p50_us <= r.p99_us && r.p99_us <= r.max_us);
    assert!(r.realtime_factor > 1.0, "{r}");
    assert_eq!(r.latency_ms, 20.0);
}

/// DFN3 outputs digital silence for noise alone and passes clean speech untouched. Its gain on
/// *synthetic* noisy speech is not a fair quality measure (it reshapes the synthetic harmonics
/// it does not recognise as voice), so quality is left to the real-speech harness in tools/eval.
#[cfg(feature = "dfn3")]
#[test]
fn dfn3_removes_noise_keeps_speech_and_limits_natively() {
    let speech = speech_like(SR * 4, 5);
    let mut chain = Chain::with_model(Model::DeepFilterNet3).unwrap();
    chain.set_max_attenuation_db(f32::INFINITY);
    let out = run_chain(&mut chain, &speech, 480);
    let lat = chain.latency();
    let kept = db(rms(&out[SR + lat..]) / rms(&speech[SR..speech.len() - lat]));
    assert!(
        kept > -2.0,
        "DFN3 lost {kept:.1} dB of clean speech-like signal"
    );

    let pink = pink_noise(0.003, SR * 4, 12);
    let reference = high_passed(&pink);
    for limit in [12.0, 24.0, f32::INFINITY] {
        let mut chain = Chain::with_model(Model::DeepFilterNet3).unwrap();
        chain.set_max_attenuation_db(limit);
        let out = run_chain(&mut chain, &pink, 480);
        let r = -db(rms(&out[2 * SR..]) / rms(&reference[2 * SR..]));
        if limit.is_finite() {
            assert!(
                r <= limit + 0.5 && r > limit - 3.0,
                "limit {limit}: attenuated {r:.1} dB"
            );
        } else {
            assert!(r > 40.0, "unlimited: attenuated only {r:.1} dB");
        }
    }
}

/// Prints the bake-off numbers. `cargo test -p oto-dsp --release [--features dfn3] -- --ignored --nocapture bench_all`
#[test]
#[ignore]
fn bench_all() {
    for model in Model::ALL.into_iter().filter(|m| m.available()) {
        println!("{}", bench(model, 20.0));
    }
}

#[test]
fn biquad_recovers_from_non_finite_input() {
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let mut f = filters::Biquad::high_pass(SR as f64, 80.0, 0.707);
        let mut clean = f.clone();
        for i in 0..100 {
            f.tick((i as f64 * 0.1).sin());
        }
        assert_eq!(f.tick(bad), 0.0);
        // From here it behaves like a freshly reset filter.
        for i in 0..1000 {
            let x = (i as f64 * 0.1).sin();
            assert_eq!(f.tick(x), clean.tick(x));
        }
    }
}

#[test]
fn sanitize_zeroes_non_finite_and_clamps() {
    let mut x = [0.5, f32::NAN, f32::INFINITY, f32::NEG_INFINITY, -1e30, 1e30];
    sanitize(&mut x);
    assert_eq!(x, [0.5, 0.0, 0.0, 0.0, -MAX_INPUT, MAX_INPUT]);
}

/// One NaN or infinity from a glitching device must not silence the chain for good.
#[test]
fn full_chain_recovers_after_nan_and_inf() {
    for enabled in [true, false] {
        let input = noisy_speech(SR * 4, 5);
        let mut reference = Chain::with_model(Model::RNNoise).unwrap();
        reference.set_enabled(enabled);
        let want = run_chain(&mut reference, &input, 256);

        let mut bad = input.clone();
        bad[SR] = f32::NAN;
        bad[SR + SR / 2] = f32::INFINITY;
        bad[SR + SR / 2 + 1] = f32::NEG_INFINITY;
        bad[SR + SR / 2 + 2] = f32::MAX;
        let mut chain = Chain::with_model(Model::RNNoise).unwrap();
        chain.set_enabled(enabled);
        let got = run_chain(&mut chain, &bad, 256);

        assert!(got.iter().all(|s| s.is_finite()), "output stays finite");
        let c = chain.limiter.ceiling();
        assert!(
            got.iter().all(|s| s.abs() <= c),
            "output stays under the ceiling"
        );
        // Half a second after the last bad sample the chain is back to normal audio.
        let tail = 2 * SR + SR / 2..;
        let (g, w) = (rms(&got[tail.clone()]), rms(&want[tail]));
        assert!(w > 1e-3, "reference has signal ({w})");
        assert!(
            db(g / w).abs() < 1.0,
            "enabled={enabled}: level after recovery {g} vs reference {w}"
        );
    }
}
