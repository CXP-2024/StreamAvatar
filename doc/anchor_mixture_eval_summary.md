# Anchor Mixture Evaluation Summary

Date: 2026-05-29

This note summarizes the current 60s `input_image_audio` comparison among:

- `teacher`
- `real_anchor_student`
- `noise+real_anchor_student`

The latest metric JSON is:

```text
outputs/teacher_real_noise_anchor_metric_comparison_latest.json
```

## Metrics

- `LSE-D`: SyncNet audio-video embedding distance. Lower is better.
- `LSE-C`: SyncNet confidence. Higher is better.
- `mouth_audio_corr`: correlation between mouth opening and audio energy. Higher is better.
- `mouth_open_std` / `mouth_velocity_mean`: visible mouth motion strength.
- `motion MSE/L1`: student motion latent error against teacher. Lower is better.
- `vel_mag_ratio`: student/teacher motion velocity magnitude ratio. Closer to `1.0` is better.

## Results

| model | LSE-D ↓ | LSE-C ↑ | mouth corr ↑ | mouth velocity | motion MSE ↓ | vel ratio |
|---|---:|---:|---:|---:|---:|---:|
| teacher | **7.679** | **4.987** | **0.305** | **0.0346** | - | - |
| real anchor student | 8.853 | **4.054** | 0.225 | 0.0276 | **0.0257** | 0.678 |
| noise+real anchor student | **8.721** | 4.036 | **0.238** | **0.0305** | 0.0277 | **0.774** |

## Takeaway

The teacher remains the strongest overall baseline.

Between the two students, the `noise+real_anchor_student` currently looks more promising for visible lip/mouth behavior:

- slightly better SyncNet `LSE-D`;
- stronger mouth motion;
- better speech/silent mouth-motion separation;
- velocity magnitude closer to teacher.

The `real_anchor_student` is still better if the objective is strict latent imitation:

- lower motion MSE;
- lower motion L1;
- slightly higher SyncNet confidence.

Current practical conclusion:

```text
noise+real anchor is better for visible video/mouth dynamics,
while real anchor is better for conservative teacher-latent matching.
```

More reliable judgment should use a multi-sample evaluation set instead of only the current `person1 + test_audio_60s` case.

## Additional Person2 Result

After re-running verification on `person2 + test_audio_60s`, the metric JSON is:

```text
outputs/teacher_real_noise_anchor_metric_comparison_person2_latest.json
```

| model | LSE-D ↓ | LSE-C ↑ | mouth corr ↑ | mouth velocity | motion MSE ↓ | vel ratio |
|---|---:|---:|---:|---:|---:|---:|
| teacher | **7.080** | **6.235** | 0.051 | 0.0326 | - | - |
| real anchor student | **8.309** | **4.027** | 0.0328 | 0.0336 | **0.0305** | 0.760 |
| noise+real anchor student | 8.429 | 3.280 | **0.0789** | **0.0359** | 0.0384 | **0.832** |

This result is more mixed than the `person1` case:

- SyncNet favors `real_anchor_student`.
- Landmark mouth-motion strength favors `noise+real_anchor_student`.
- Latent MSE/L1 favors `real_anchor_student`.
- Velocity magnitude ratio favors `noise+real_anchor_student`.

Updated conclusion:

```text
noise+real anchor consistently increases visible motion strength,
but SyncNet lip-sync quality is not consistently better across references.
real anchor remains the more conservative and stable teacher-latent imitation.
```

The next reliable comparison should average SyncNet and landmark metrics over multiple reference images/audio clips.
