# FM-Style One-Step Motion Denoiser Architecture

This note describes the current StreamAvatar/DyStream student architecture used
for fast streaming motion generation.

## Goal

The goal is to replace the slow frame-level AR + FM teacher path with a faster
student that can generate short blocks of motion latents directly:

```text
audio chunk + motion history + reference anchor + noisy future tokens
  -> student
  -> clean future motion latents
  -> DyStream renderer
  -> video frames
```

The current version is not a full flow-matching ODE generator. It is better
described as an FM-style one-step conditional denoiser.

## Standard Flow Matching Reference

A standard flow-matching generator usually works like this:

```text
x_T ~ Gaussian noise

for t = T ... 0:
    v_t = model(x_t, t, condition)
    x_{t-dt} = ODE_step(x_t, v_t, t)

return x_0
```

The model predicts a velocity field. During inference, the scheduler integrates
this field through multiple ODE steps to reach a clean sample.

That is the classical FM interpretation:

```text
noisy state -> velocity field -> ODE integration -> clean motion
```

## Current Student Formulation

Our current `cross_fm` student keeps the noise-conditioned idea, but uses a
direct clean-motion objective.

During training:

```text
clean future motion block x_0
noise epsilon ~ N(0, I)
timestep t ~ scheduler timesteps

x_t = FlowMatchEulerDiscreteScheduler.scale_noise(
    sample=x_0,
    timestep=t,
    noise=epsilon,
)

student(
    past_motion,
    audio_self,
    audio_other,
    anchor,
    noisy_future=x_t,
    timestep=t,
) -> pred_clean_motion

loss = MSE(pred_clean_motion, x_0)
     + velocity loss
     + acceleration loss
     + boundary loss
```

So the noise level is controlled by the scheduler timestep. Conceptually this is
similar to a linear interpolation between clean motion and Gaussian noise:

```text
x_t ~= (1 - sigma_t) * x_0 + sigma_t * epsilon
```

The exact coefficient is determined by `FlowMatchEulerDiscreteScheduler`.

## Inference Behavior

During inference, the current model does not run an ODE loop.

Instead:

```text
noisy_future ~ Gaussian noise
t = scheduler.timesteps[0]

pred_clean_motion = student(
    past_motion,
    audio_self,
    audio_other,
    anchor,
    noisy_future,
    t,
)
```

The predicted clean motion block is then appended to the rolling motion history
and sent to the renderer.

This means the student is trained and used as a one-step denoiser:

```text
noisy future block + condition -> clean future block
```

It is FM-inspired because it uses a flow-matching scheduler, noisy states, and
timestep conditioning. It is not a strict velocity-field FM model because it does
not predict velocity and does not integrate with an ODE solver at inference.

## Data Flow

```text
Audio waveform
  |
  | frozen DyStream wav2vec/audio encoders
  v
audio_self, audio_other features

Reference image / first motion latent
  |
  v
anchor motion latent

Previous generated/GT motion latents
  |
  v
past_motion history

Gaussian future noise + timestep
  |
  v
noisy_future tokens

audio features + anchor + history + noisy future
  |
  v
BlockCrossFMStudent
  |
  v
future motion latents
  |
  v
DyStream decoder / renderer
  |
  v
video frames
```

## Training Targets

Two target sources are supported:

```text
teacher target:
  frozen DyStream teacher rollout -> teacher motion sequence

cache target:
  LRS3 video frames -> DyStream visual motion encoder -> motion_latent
```

The cache-target path is closer to supervised training on real video motion
latents. The teacher-target path is classical distillation from the original
DyStream motion generator.

In both cases, the student architecture and inference interface are the same.

## Why This Design

This design is a compromise between quality and speed.

Advantages:

- one student forward can emit multiple future motion frames;
- inference avoids multi-step ODE sampling;
- noisy future tokens prevent the model from being a purely deterministic
  history/audio regressor;
- timestep conditioning gives a path toward future FM/consistency distillation;
- rolling history helps long-video identity and motion consistency.

Limitations:

- the model currently predicts clean motion, not FM velocity;
- there is no scheduler integration during inference;
- noise level at inference must match the training distribution well;
- if the model overuses history, it can become under-moving or slow;
- cache-target quality depends on crop/alignment quality before motion encoding.

## Difference From Pure Regression

A pure block regressor would use:

```text
past_motion + audio + anchor -> future_motion
```

The current model adds:

```text
noisy_future + timestep
```

This makes the task closer to conditional denoising. It is still one-step and
clean-target supervised, but the student must learn how clean motion lives under
different noise levels, instead of only learning a direct deterministic mapping.

## Upgrade Path To Full FM

A stricter FM version would change the target from clean motion to velocity:

```text
x_t = interpolate(x_0, noise, t)
target_velocity = flow_target(x_0, noise, t)

student(..., x_t, t) -> pred_velocity
loss = MSE(pred_velocity, target_velocity)
```

Inference would then use:

```text
x = Gaussian noise
for t in scheduler.timesteps:
    v = student(..., x, t)
    x = scheduler.step(v, t, x)
return x
```

This is more principled but slower. A practical next step is consistency or
few-step distillation:

```text
teacher multi-step FM result -> student one-step or two-step result
```

That would keep the FM formulation while preserving most of the speed advantage.

## Current Position

The current architecture should be named and discussed as:

```text
FM-style blockwise one-step conditional motion denoiser
```

It is suitable as the fast streaming student in the current project. If the
paper direction requires stricter generative modeling claims, the next version
should move from clean-motion denoising to velocity-field prediction plus
one-step/few-step consistency distillation.
