# Blockwise AR Denoising Student Architecture

This note records the actual student architecture used by the current
`cross_fm` configuration. The name `cross_fm` is historical and slightly
misleading: the student does not reproduce the teacher's AR + FM ODE sampler.
It is a blockwise autoregressive denoising Transformer.

## Short Answer

The middle Transformer block is:

```text
LayerNorm
-> prefix self-attention over [history tokens, noisy future tokens]
-> residual add
-> LayerNorm
-> audio cross-attention
-> residual add
-> LayerNorm
-> anchor cross-attention
-> residual add
-> LayerNorm
-> FFN
-> residual add
```

This whole residual block is repeated `L` times, where `L = cfg.student.layers`
in the config. In the current 60k pretrain teacher-cache-anchor config, `L = 6`.

So the accurate block label is:

```text
(prefix self-attention -> audio cross-attention -> anchor cross-attention -> FFN) x L
```

It should not be drawn as a separate AR module followed by a separate FM head.
It also should not be drawn as a single combined `audio / anchor cross-attention`
if we want to match the code exactly, because audio and anchor are two separate
cross-attention sublayers.

## Inputs

For each block, the student receives:

```text
past_motion:   (B, H, 512)
audio_self:    (B, H + K, 768)
audio_other:   (B, H + K, 768)
anchor:        (B, 1, 512)
noisy_future:  (B, K, 512)
timestep:      (B,)
```

Current common config:

```text
H = history_frames = 32
K = block_frames = 2
motion_dim = 512
hidden_dim = 512
audio_dim = 768
```

## Two Separate Streams

The factual diagram should show two separate input streams.

### 1. Token Stream

Only motion history, noisy future, and timestep become Transformer sequence
tokens:

```text
past_motion
  -> motion_proj
  -> history tokens

noisy_future
  -> noisy_future_proj
  -> future tokens

timestep
  -> time_proj
  -> added to future tokens only

[history tokens, future tokens]
  + type embedding
  + positional embedding
  -> Transformer sequence x
```

This is implemented in `BlockCrossFMStudent.forward()`:

```text
motion_tokens = motion_proj(past_motion)
future_tokens = noisy_future_proj(noisy_future)
future_tokens = future_tokens + time_proj(timestep)
x = cat([motion_tokens, future_tokens], dim=1)
x = x + type_embed + pos_embed
```

### 2. Condition Memory Stream

Audio and anchor do not go into the Token Builder. They become cross-attention
memories:

```text
audio_self, audio_other
  -> audio_self_proj, audio_other_proj
  -> concat
  -> audio_fusion
  -> audio_memory

anchor
  -> anchor_proj
  -> anchor_memory
```

Then each Transformer residual block attends to those memories with two
separate cross-attention calls:

```text
x attends to audio_memory
x attends to anchor_memory
```

## Prefix Self-attention Mask

The self-attention mask is:

```python
mask = zeros(H + K, H + K)
mask[:H, H:] = True
```

This means:

- history query tokens cannot attend to future tokens;
- future query tokens can attend to history tokens and the future-token block.

So "prefix self-attention" is a better label than plain causal self-attention.
It protects the history prefix from future leakage, while allowing the future
block to be predicted jointly.

## Rollout Behavior

The student is autoregressive across blocks, not inside the teacher's original
per-frame AR+FM loop.

For each block:

```text
1. Take current history: past = last H motion frames.
2. Gather audio window of length H + K.
3. Create or sample noisy_future of length K.
4. Run one student forward pass.
5. Output K clean motion latents.
6. Append predicted block to history.
7. Continue to the next block.
```

In code this is `blockwise_fm_rollout()`. Despite the name, it does not run an
ODE solver. It calls the student once per block.

## Training Versus Inference

### Training

During training, if target motion is available, the clean target block is
converted into `noisy_future`:

```text
clean target block + scheduler noise -> noisy_future
```

The student then predicts the clean motion block directly.

Important: the loss is not a velocity-field FM loss. It is supervised motion
matching against the teacher or cached target sequence, plus temporal losses.

Current loss:

```text
loss_motion   = MSE(student_seq, target_seq)
loss_velocity = MSE(delta student, delta target)
loss_acc      = MSE(second-delta student, second-delta target)
loss_boundary = MSE(block-boundary jump student, target)

loss = loss_motion
     + lambda_v * loss_velocity
     + lambda_a * loss_acc
     + lambda_b * loss_boundary
```

### Inference

At inference, there is no target future block. The student receives Gaussian
future noise:

```text
noisy_future = randn(B, K, 512)
timestep = scheduler.timesteps[0]
student forward once
-> clean motion block
```

There is no iterative denoising loop inside the student. The design is better
described as a one-step denoising student with blockwise AR rollout.

## What The Figure Should Show

The corrected figure should use this layout:

```text
                        token stream
motion history  ----\
future noise+t -----> Token Builder ----\
                                         \
                                          -> Transformer blocks x L -> future-position head -> clean block
                                         /
                        condition memory/
audio features ----\
reference anchor --> Memory Builder ----/

clean block -> append to history -> next block
```

Inside `Transformer blocks x L`, the accurate sublayers are:

```text
prefix self-attention
audio cross-attention
anchor cross-attention
feed-forward residual
```

The figure should avoid showing:

- audio/reference entering the Token Builder;
- a separate FM head after the Transformer;
- an ODE sampler inside the student;
- a velocity-field prediction target;
- a single fused audio+anchor attention if the goal is code-level accuracy.

## Practical Naming

Recommended names:

- "Blockwise AR Denoising Student"
- "One-step future-token denoising student"
- "Blockwise cross-attention denoiser"

Avoid using "Cross-FM student" in the report unless it is explicitly explained
as a historical config name. The current implementation is FM-like only in the
use of noisy future tokens and timestep conditioning, not in the teacher's full
flow-matching sampler sense.
