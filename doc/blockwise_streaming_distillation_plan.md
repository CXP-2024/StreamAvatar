# Block-wise Streaming AR Distillation Plan

## 目标

当前 DyStream 的主要瓶颈已经不是 FM head，而是 GPT/AR 部分。原推理流程是逐帧自回归：

```text
for each frame:
    GPT(history motion + audio + anchor) -> current condition
    FM(condition + noise + timestep) -> current motion latent
    append current motion latent to history
```

`chunk_frames=8` 目前只是把 8 个逐帧 AR step 包在一次调用里返回，并没有让 GPT 一次并行预测 8 帧。因此它主要减少 app 调度和 renderer batch 开销，对 GPT 本身加速有限。

本方案的目标是把 frame-level AR 改成 block-level streaming AR：

```text
for each audio chunk:
    BlockAR student(history motion + K-frame audio + anchor)
        -> K motion latents in one forward
    append K generated latents to history
```

这会真正减少 GPT/AR forward 次数，并且比单纯 KV-cache 更像一个模型方法。

## 创新点定位

KV-cache 是合理工程优化，但单独作为课程项目创新较弱。更合适的主线是：

```text
Streaming Block Autoregressive Distillation
```

核心变化：

- 生成单位从 1 frame 变成 K-frame block。
- teacher 是原 DyStream frame-by-frame AR rollout。
- student 是一次输出 K 帧的 streaming block predictor。
- 用 continuity-aware loss 保证 block 边界稳定。

可以做清晰 ablation：

```text
K = 1, 2, 4, 8, 16
```

比较：

- latency
- real-time factor
- latent MSE to teacher
- velocity / acceleration error
- block boundary discontinuity
- rendered visual quality

## 与原 DyStream 的关系

原 DyStream：

```text
audio waveform
  -> Wav2Vec2 encoder
  -> Audio2FaceGPT autoregressive transformer
  -> Flow Matching head
  -> 512-D motion latent
  -> pretrained renderer
```

本方案第一版 student：

```text
audio waveform
  -> frozen DyStream Wav2Vec2 encoder
  -> audio features
  -> BlockARStudent(history motion + future audio features + anchor)
  -> K x 512-D motion latents
  -> pretrained renderer
```

第一版先绕开 FM head，直接蒸馏 teacher 的 final motion latent。原因：

- 当前 profiling 中 FM 不是瓶颈。
- 直接预测 motion latent 能最大幅度减少 GPT+FM 逐帧循环。
- 训练目标更简单，便于快速验证 block-wise 生成是否稳定。

后续如果直接 latent regression 不够稳定，可以再改成：

```text
BlockARStudent -> K GPT conditions
shared FM head parallel denoise K frames
```

## 模型输入输出

### 输入

```text
past_motion:       [B, H, 512]
past_audio_feat:   [B, H, 768]
future_audio_feat: [B, K, 768]
anchor_motion:     [B, 1, 512]
```

其中：

- `H` 是历史 motion window，例如 32。
- `K` 是一次生成帧数，例如 8。
- audio feature 来自冻结 DyStream Wav2Vec2，已经对齐到 25 fps。

### 输出

```text
pred_motion_block: [B, K, 512]
```

预测出的 K 帧会 append 到 history，作为下一次 block 的 past motion。

## Teacher

teacher 使用原始 DyStream checkpoint：

```text
teacher.inference(
    denoising_steps = 1 or 5,
    guidance_mode = full_5way or all_only
)
```

建议两组 teacher：

1. `all_only + ODE=1`：和当前实时 baseline 完全一致，训练最直接。
2. `full_5way + ODE=5`：质量更强，但 student 学起来更难。

第一版推荐从 `all_only + ODE=1` 开始，因为目标是加速当前已经可接受的 stream app。

## Loss

基础 loss：

```text
L_motion = MSE(x_student, x_teacher)
```

连续性 loss：

```text
L_vel = MSE(delta(x_student), delta(x_teacher))
L_acc = MSE(delta2(x_student), delta2(x_teacher))
```

边界 loss：

```text
L_boundary = MSE(x_student_block_first - previous_student_last,
                 x_teacher_block_first - previous_teacher_last)
```

总 loss：

```text
L = L_motion
  + lambda_vel * L_vel
  + lambda_acc * L_acc
  + lambda_boundary * L_boundary
```

第一版默认：

```text
lambda_vel = 0.5
lambda_acc = 0.1
lambda_boundary = 0.2
```

## 训练方式

每个 batch：

```text
1. 读取一段 audio。
2. 用冻结 teacher 的 Wav2Vec2 生成 audio features。
3. 用 teacher 原始 AR inference 生成 teacher motion sequence。
4. student 从 anchor repeated history 开始 block-wise rollout。
5. 每次预测 K 帧，append 到自己的 history。
6. 对整段 student motion 和 teacher motion 计算 loss。
7. 只更新 BlockARStudent。
```

注意这里不是 teacher forcing。student 的后续 block 会看到自己前面生成的 motion，因此训练分布更接近真实流式推理。

## 第一版实现范围

新增文件：

- `train_blockwise_distill.py`
- `configs/distill/blockwise_stream_distill.yaml`

第一版只做训练，不修改原 app 推理路径。

保存 checkpoint：

```text
outputs/blockwise_stream_distill/blockwise_last.pt
outputs/blockwise_stream_distill/blockwise_best.pt
```

checkpoint 内容：

```text
{
  "student": state_dict,
  "config": yaml config,
  "step": global_step,
  "loss": last_loss
}
```

## 后续接入计划

### Phase 1: 训练脚本 smoke test

命令：

```bash
.venv/bin/python train_blockwise_distill.py \
  --config configs/distill/blockwise_stream_distill.yaml \
  --override data.max_clips=10 training.max_steps=100
```

目标：

- loss 能正常下降。
- checkpoint 能保存。
- student 输出 shape 正确。

### Phase 2: 离线 verify

新增 verify 脚本：

```text
teacher video vs blockwise student video
```

横向 concat 对比：

```text
teacher | student K=4 | student K=8
```

### Phase 3: 接入 stream app

新增一个模式：

```text
motion_backend = original_ar | blockwise_student
```

上传图片后：

- 预处理 reference image。
- 加载 teacher audio encoder / renderer。
- 加载 blockwise student。

录音 chunk 到达后：

- 计算或复用 audio features。
- `student.stream_step(audio_features_chunk, state)`。
- batch render chunk。

### Phase 4: ablation

实验组合：

```text
K = 1, 2, 4, 8, 16
history = 16, 32, 64
teacher = all_only_1step, full_5way_5step
```

## 风险

### 1. 嘴型下降

K 越大，并行预测越多，帧间细节依赖越弱。快速音素可能变糊。

缓解：

- K 从 4 开始。
- 加 velocity / acceleration loss。
- 加局部 mouth landmark 或 SyncNet loss，后续再做。

### 2. 头动漂移

student 自己预测历史，可能逐 block 漂移。

缓解：

- boundary loss。
- anchor conditioning。
- 每 N 个 chunk 做 weak re-anchor。

### 3. Teacher cache 成本

在线生成 teacher target 会慢。

缓解：

- 第一版在线 teacher 方便验证。
- 后续将 teacher motion latent 离线缓存。

## 近期判断

如果目标是快速形成课程项目创新，推荐顺序：

```text
1. block-wise distill smoke test
2. K=4/K=8 离线视频对比
3. 接入 stream app
4. 做速度/质量 ablation
5. 再考虑 KV-cache 作为工程 baseline
```
