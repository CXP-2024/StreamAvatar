# DyStream Training and Distillation Notes

这份文档整理三个问题：

1. 原 DyStream 模型/代码里，motion model 是怎么训练的。
2. GPT、Flow Matching head、audio encoder、renderer 这些组件之间的训练与冻结关系。
3. 如果目标是加速到流式形态，当前架构更适合做哪些蒸馏或工程优化。

参考来源：
- DyStream project/repo: https://github.com/RobinWitch/DyStream
- DyStream paper entry: https://arxiv.org/abs/2512.24408
- Flow Matching: https://arxiv.org/abs/2210.02747
- Progressive Distillation: https://arxiv.org/abs/2202.00512
- Rectified Flow / Reflow: https://arxiv.org/abs/2209.03003
- Consistency Models: https://arxiv.org/abs/2303.01469
- Latent Consistency Models: https://arxiv.org/abs/2310.04378

## 结论先说

DyStream 的核心不是端到端从像素训练一个 talking-head 模型，而是把问题拆成：

```text
audio waveform
  -> Wav2Vec2 audio encoder
  -> Audio2FaceGPT autoregressive transformer
  -> Flow Matching diffusion head
  -> 512-D motion latent sequence
  -> pretrained renderer / face generator
  -> video frames
```

当前代码里的主训练对象是 `Audio2FaceGPT` 整个 motion generator，包括：

- 两路 Wav2Vec2 audio encoder。
- audio projection / fusion。
- 12 层 autoregressive GPT blocks。
- anchor embedding / face embedding / output projection。
- 6 层 `DiffusionHead`。

只有参数名里包含 `freeze` 的模块会被 `main.py` 自动冻结。当前实际使用的 `motion_gen_gpt_flowmatching_addaudio_linear_twowavencoder.py` 里，两路 `WrapedWav2Vec` 没有 `freeze` 命名，所以如果从头按 `main.py` 训练，它们理论上也是可训练的；实际是否充分更新取决于原作者训练配置、resume checkpoint 和 learning rate。

渲染器、参考图处理、motion encoder / face generator 是单独的可视化/渲染模型，不在 `main.py` 的 motion training loss 里训练。它们是预训练组件，用于把 motion latent decode 成图片。

## 原仓库公开信息

README 中说明项目在逐步开源，offline generation 和 Gradio demo 已开源，training code 仍标注为 TODO。不过我们当前仓库已经包含 `main.py`、dataset 和 config，能看到 motion generator 的实际训练逻辑。

因此最可靠的训练细节来自代码：

- [main.py](/mnt/pfs/group-jt/changxun.pan/runs/test/float_playground/DyStream/main.py:85)
- [datasets/single_dyadic_prev_audio.py](/mnt/pfs/group-jt/changxun.pan/runs/test/float_playground/DyStream/datasets/single_dyadic_prev_audio.py:1)
- [configs/motion_gen/sample.yaml](/mnt/pfs/group-jt/changxun.pan/runs/test/float_playground/DyStream/configs/motion_gen/sample.yaml:1)
- [model/motion_generation/motion_gen_gpt_flowmatching_addaudio_linear_twowavencoder.py](/mnt/pfs/group-jt/changxun.pan/runs/test/float_playground/DyStream/model/motion_generation/motion_gen_gpt_flowmatching_addaudio_linear_twowavencoder.py:339)

## 训练数据形式

dataset 每次取一个固定长度 window：

```text
motion_latent:       [B, 96, 512]
audio:               [B, 96 * 640]  # 25 fps, 16 kHz, 每帧 640 audio samples
audio_other:         [B, 96 * 640]
style_latent:        reference motion latent sequence
style_latent_other:  other speaker/listener reference motion latent sequence
```

关键配置：

- `pose_fps = 25`
- `audio_sr = 16000`
- `cbh_window_length = 96`
- `motion_dim = 512`
- `wav2vec_layer = 8`
- validation 默认 `denoising_steps = 5`

dataset 读的是已经预处理好的 motion latent，不直接从原视频帧训练 renderer。`motion_self_path` / `motion_other_path` 是 `.npz`，里面的 `random_data` 是 `[T, 512]` latent 序列。

## 组件关系

### 1. Wav2Vec2 audio encoder

代码位置：

- [model/motion_generation/motion_gen_gpt_flowmatching_addaudio_linear_twowavencoder.py](/mnt/pfs/group-jt/changxun.pan/runs/test/float_playground/DyStream/model/motion_generation/motion_gen_gpt_flowmatching_addaudio_linear_twowavencoder.py:232)

结构：

```text
WrapedWav2Vec
  Wav2Vec2Model.from_pretrained(local wav2vec2-base-960h)
  feature_extractor
  feature_projection
  first 8 Wav2Vec2 encoder layers
  attention patched to causal attention
```

有两路实例：

- `audio_encoder_face`
- `audio_encoder_face_other`

它们分别处理 self audio 和 other audio。输出约为 50 fps，然后插值到 25 fps，对齐 motion latent。

冻结关系：

- Wav2Vec2 权重初始化来自 `pretrained_model/wav2vec2-base-960h`。
- 当前 `main.py` 只按名字包含 `freeze` 来冻结参数。
- 当前实际模型里的 `audio_encoder_face` / `audio_encoder_face_other` 名字不含 `freeze`，所以默认会参与训练。
- 但从加速/蒸馏角度看，不建议继续训练 audio encoder。它计算量不是主瓶颈，微调风险大，冻结更稳。

### 2. Audio2FaceGPT autoregressive backbone

代码位置：

- [model/motion_generation/motion_gen_gpt_flowmatching_addaudio_linear_twowavencoder.py](/mnt/pfs/group-jt/changxun.pan/runs/test/float_playground/DyStream/model/motion_generation/motion_gen_gpt_flowmatching_addaudio_linear_twowavencoder.py:339)

实际结构来自 Python 默认参数，不完全等于 yaml 里的 `hidden_size/layers/heads`：

```text
hidden_size = 768
num_layers = 12
num_heads = 12
mlp_ratio = 4
face_dim = 512
```

GPT block 输入：

```text
face_hidden:   previous/noisy motion latent embedded to 768
audio_hidden:  self+other audio fused condition, 768
anchor_hidden: reference motion / anchor latent, 768
causal mask:   autoregressive temporal mask
```

训练时的 GPT 不是自由 rollout，而是 teacher-forcing 风格：输入 `face_latent_gt[:, :-1]` 的 noisy/masked 版本，预测后续位置的 latent 条件。

推理时则是真正 autoregressive：

```text
past_motion -> GPT predicts current condition -> FM head denoises current motion
current motion appended to history -> next frame
```

这就是为什么我们后面做蒸馏时，单纯一帧 MSE 很容易过拟合得很好，但 rollout 视频会崩：训练分布和推理分布不一致。

### 3. Flow Matching diffusion head

代码位置：

- [model/motion_generation/motion_gen_gpt_flowmatching_addaudio_linear_twowavencoder.py](/mnt/pfs/group-jt/changxun.pan/runs/test/float_playground/DyStream/model/motion_generation/motion_gen_gpt_flowmatching_addaudio_linear_twowavencoder.py:285)

结构：

```text
DiffusionHead
  noisy_proj: 512 -> 768
  gpt_proj:   512 -> 768
  6 x DiffusionBlock
  output_proj: 768 -> 512
```

每个 `DiffusionBlock` 是 MLP + AdaLN 调制，不是 attention block。它接收：

- 当前 noisy latent。
- GPT 输出的 condition。
- timestep embedding。

训练时采样随机 timestep，并用 `FlowMatchEulerDiscreteScheduler.scale_noise()` 得到 noisy latent。

一个容易误解的点：当前代码的 loss 不是显式 velocity MSE。`main.py` 里 `motion_pred` 直接和 clean `motion_latent[:, 1:]` 做 MSE：

```text
loss = MSE(model_pred, clean_motion_latent)
```

推理时，scheduler 需要 velocity，所以代码把 diffusion head 输出当作 denoised latent，再转换：

```text
velocity = (latent_t - noise_pred) / sigma
latent_t = scheduler.step(velocity, timestep, latent_t)
```

所以它是 Flow Matching scheduler 框架下的 x0/clean-latent prediction 形式，而不是直接监督 `v = x1 - x0` 的 velocity head。

### 4. Renderer / visualization model

代码位置：

- [app.py](/mnt/pfs/group-jt/changxun.pan/runs/test/float_playground/DyStream/app.py:143)

作用：

```text
reference image
  -> crop/resize/mask
  -> motion_encoder extracts reference 512-D latent
  -> motion generator predicts driving 512-D latent sequence
  -> flow_estimator(reference_latent, driving_latent)
  -> face_generator(flow, reference features)
  -> video frames
```

冻结关系：

- renderer 不在 `main.py` 的 optimizer 里。
- 它通过 visualization checkpoint 加载，用于 image preprocessing、motion latent extraction、latent-to-frame rendering。
- 这部分是预训练组件，不是当前 motion generation loss 的训练对象。

## 原训练过程

从当前代码看，motion generator 的训练流程是单阶段的 latent denoising / flow-matching style training：

```text
1. 读取一个 96-frame clip 的 clean motion latent x0。
2. 采样 Gaussian noise。
3. 随机采样 flow scheduler timestep t。
4. 用 scheduler.scale_noise(x0, noise, t) 得到 x_t。
5. 生成 mask，把一部分历史位置替换成 noisy latent，一部分保留 clean latent。
6. 叠加一个从前到后的 linear ramp，使靠后的 motion 更接近 clean。
7. 从最后 10 帧随机取一个 anchor motion。
8. Audio2FaceGPT 接收：
   - masked/noisy face latent history
   - full noisy latent
   - timestep
   - self audio
   - other audio
   - anchor latent
9. 输出 [B, 95, 512] motion prediction。
10. 和 clean motion_latent[:, 1:] 做 weighted MSE。
11. 最后 5 帧 loss 权重更高。
12. EMA 跟踪模型参数，用于验证/推理。
```

loss 形式：

```text
per_frame_loss = mean_dim512((pred_motion - clean_motion)^2)
weights = 1
weights[last 5 frames] = 3
loss = sum(per_frame_loss * weights) / sum(weights)
```

这里没有 pixel loss、perceptual loss、sync loss、landmark loss。训练目标完全在 512-D motion latent 空间。

## CFG 训练与推理

训练时通过 condition dropout 支持 classifier-free guidance：

- `drop_audio`
- `drop_audio_other`
- `drop_anchor`

推理时原始模式会构造 5 路条件：

```text
0: no audio, no other audio, no anchor
1: anchor only
2: self audio only
3: other audio only
4: self audio + other audio + anchor
```

再按 CFG 权重组合：

```text
guided = uncond
       + cfg_audio       * (self_audio - uncond)
       + cfg_audio_other * (other_audio - uncond)
       + cfg_anchor      * (anchor - uncond)
       + cfg_all         * (all - uncond)
```

我们在 stream app 里加入过 `all_only`，就是只跑第 4 路全条件。它能省掉 CFG 分支计算，但实际提速有限，因为 GPT 自回归窗口重复计算仍然是主瓶颈。

## 当前瓶颈判断

我们在 B200 上的 stream app profiling 看到过类似结果：

```text
motion_inference:      ~3.5s - 5.1s
motion_gpt:            ~2.6s - 2.8s
motion_fm:             ~0.4s - 0.5s
motion_audio_encoder:  ~0.02s
render:                ~1.4s - 1.7s
mux:                   ~1.0s
```

这说明：

- 现在 ODE=1 时，FM head 已经不是最大瓶颈。
- GPT autoregressive 重复跑历史窗口是最大瓶颈。
- render 是第二梯队瓶颈。
- audio encoder 几乎不是瓶颈，因为整段音频只编码一次。

所以如果当前已经能接受 ODE=1 的质量，继续做 5-step -> 1-step FM 蒸馏的收益不大。更大的收益来自：

1. GPT 增量推理 / KV cache。
2. 把 5 路 CFG 蒸馏进 all-only。
3. rollout-aware student，让学生在真实推理分布下稳定。
4. 减少生成帧率或 chunk 计算量。

## 适合当前架构的蒸馏路线

### 路线 A：FM step distillation

目标：

```text
teacher: NFE=5 或更多
student: NFE=1
```

适用场景：

- 原始 5-step 质量明显好于 1-step。
- FM head 是瓶颈。
- 希望 student 一步达到 teacher 多步效果。

可用方法：

- Progressive Distillation：逐步把 5 step 压到 3 step，再压到 1 step。
- Consistency Distillation：同一 ODE/flow 轨迹上的不同时间点输出应一致。
- Rectified Flow Reflow：用 teacher 轨迹重新构造更直的 flow，让采样路径更短。

但对我们当前情况，优先级不高，因为 ODE=1 已经是默认可用配置，而且 profiling 里 `motion_fm` 只占约 0.4s。

### 路线 B：CFG branch distillation

目标：

```text
teacher: full_5way CFG output
student: all_only output
```

训练方式：

```text
同一个 audio/reference/history/noise:
  teacher_motion = full_5way inference(...)
  student_motion = all_only inference(...)
  loss = MSE(student_motion, teacher_motion)
       + velocity smoothness loss
       + acceleration smoothness loss
```

优点：

- 不改变模型主结构。
- 能让 all-only 学到 CFG 后的 guided distribution。
- 对 stream app 简单，推理继续使用 `guidance_mode=all_only`。

缺点：

- 我们已测过只关 CFG 提速不巨大，所以它更多是质量稳定路线，不是最大加速路线。

### 路线 C：AR rollout-aware distillation

这是当前最值得做的训练路线。

目标：

```text
teacher: 原模型，full quality 或当前可接受设置
student: 流式推理设置，例如 ODE=1 + all_only + chunk_frames=8
```

不要只监督单帧 teacher-forcing 输出，而要监督自由 rollout：

```text
for t in streaming chunks:
    student 用自己前面生成的 motion 作为 history
    teacher 用原始模型生成同一段 target motion
    loss 回传到整个 chunk
```

推荐 loss：

```text
L_motion = MSE(x_student, x_teacher)
L_vel    = MSE(Δx_student, Δx_teacher)
L_acc    = MSE(Δ²x_student, Δ²x_teacher)
L_anchor = MSE(first generated frame, reference/local continuity)

L = L_motion + 0.5 * L_vel + 0.1 * L_acc + optional L_anchor
```

为什么重要：

- 当前真实推理是 autoregressive。
- 如果训练只用 GT history 或 teacher history，学生一旦自己生成的历史有偏差，误差会积累。
- rollout-aware 训练可以让模型在“自己产生的历史分布”上学习稳定。

这也是之前单帧/短 clip loss 很低但视频质量不稳定的根因之一。

### 路线 D：GPT incremental / KV-cache，不一定需要蒸馏

这是最直接的加速方向。

当前 `one_clip_only_inference()` 每生成一帧都会重新跑：

```text
x = face_hidden[:, :t]
for block in 12 blocks:
    x = block(x, ...)
```

也就是每一帧都重复计算历史 token。严格的 causal transformer 理论上可以缓存每层 K/V：

```text
past_key_values[layer] += current_token_key_value
只计算当前 token
```

但 DyStream 的 block 里不只有普通 self-attention，还有：

- audio linear injection。
- anchor cross attention。
- 第二个 positional self-attention。
- RoPE / sinusoidal PE 混合。

所以 KV-cache 不是一行代码能加，但它是当前最可能带来大幅提速的工程项。它不改变训练目标，也不需要重新训练。

### 路线 E：低 fps motion distillation

目标：

```text
teacher: 25 fps motion
student: 12.5 fps motion
postprocess: interpolate motion latent to 25 fps, then render
```

优点：

- 直接减少 GPT 自回归步数，理论上接近 2x。
- 对当前瓶颈更有效。

风险：

- 嘴型/lip-sync 可能变差。
- motion latent 插值不一定等价于真实中间表情。
- 需要视觉对比和音画同步评估。

这是一个可以快速试验的方向，但要作为 paper 方案需要更严谨的同步指标。

### 路线 F：小 GPT student

目标：

```text
teacher: 12-layer GPT
student: 6-layer / 4-layer GPT
```

训练：

- 初始化 student 为 teacher 的隔层权重，例如取 0,2,4,6,8,10。
- hidden state distillation。
- rollout latent distillation。
- 可选最后再 fine-tune diffusion head。

优点：

- 真正减少主瓶颈模型参数和计算。

缺点：

- 改动比 FM distill 大。
- 更容易损伤长期稳定性。
- 需要更多训练和验证。

建议在 KV-cache / low-fps / rollout-aware all-only 都做过之后再考虑。

## 当前 distill 脚本应该怎么定位

`train_distill.py` 目前是一个局部蒸馏脚本，核心思想是：

```text
teacher: 原 checkpoint，冻结
student: 原 checkpoint 初始化，冻结大部分，只训练 diffusion_head，可选 time_embed
loss: student rollout 输出 vs teacher rollout 输出
```

它适合验证 FM head 或少量模块能否适配 1-step，但不应该期待它显著降低 GPT 计算量。因为 GPT 仍然完整存在，而且推理仍然自回归。

如果要真正提升 stream app 的速度，distill 目标需要从“只替换 diffusion head”扩展到：

- all-only guided distribution。
- rollout-aware streaming distribution。
- 或减少 GPT 自回归计算本身。

## 推荐实施优先级

### P0：保留当前可用 realtime 设置

继续使用：

```text
denoising_steps = 1
guidance_mode = all_only
motion_chunk_frames = 8
render_batch = 8
```

这是当前最快能出效果的 baseline。

### P1：做真正 stream state 接口

把现在每个 chunk 的推理整理成：

```text
state = {
  past_motion,
  past_audio_tail,
  precomputed_audio_feature_tail,
  reference_latent,
  anchor_latent,
}

next_frames, state = stream_step(audio_chunk, state)
```

目标不是先提速，而是让训练/推理分布固定下来。后续所有蒸馏都应该对齐这个接口。

### P2：rollout-aware distill

teacher 用原始/高质量设置生成 target latent sequence。student 用真实 stream settings 自由 rollout。

建议先不训 audio encoder：

```text
frozen:
  audio_encoder_face
  audio_encoder_face_other
  renderer
  motion_encoder

trainable stage 1:
  diffusion_head
  time_embed

trainable stage 2:
  output_proj
  selected GPT blocks or LoRA adapters
```

这样比直接全量训练稳，也更符合我们数据量有限的情况。

### P3：KV-cache / incremental GPT

这是工程复杂但收益最大的加速项。先用 profiling 确认每个 GPT block 的耗时，再做最小 cache：

- 先 cache 第一段 self-attention。
- 再处理第二段 positional self-attention。
- anchor cross attention 可以预计算 anchor K/V。
- audio hidden 已经按帧预计算，stream step 只切当前窗口。

### P4：低 fps motion 实验

试验：

```text
生成 12.5 fps motion latent
linear/cubic interpolate 到 25 fps
render 25 fps
```

如果视觉效果可接受，这是最快把 GPT 调用数减半的方法。

### P5：FM consistency / progressive distill

只有在 1-step 明显损伤质量时再做。否则它对当前实时瓶颈帮助有限。

## 训练阶段建议

如果我们从论文/原模型思路出发，做自己的蒸馏版，可以分成以下阶段：

### Stage 0：teacher cache

用原始 checkpoint 跑一批音频/参考图，缓存：

```text
audio path
reference image / reference latent
teacher motion latent
teacher rendered video optional
timing profile
```

注意 teacher cache 必须使用与目标 stream app 相同的 crop/resize/reference latent，否则可视化会错位。

### Stage 1：只训 FM head

冻结：

```text
Wav2Vec2 encoders
GPT blocks
renderer
motion encoder
```

训练：

```text
diffusion_head
time_embed optional
```

目标：

```text
student 1-step latent ~= teacher multi-step latent
```

这个阶段主要验证 denoising head 的质量，不解决 GPT 慢。

### Stage 2：rollout-aware all-only

冻结 audio encoder 和 renderer，训练：

```text
diffusion_head
time_embed
output_proj
optional small adapters in GPT blocks
```

目标：

```text
student all-only rollout ~= teacher full-CFG rollout
```

这是最贴近当前 stream app 的蒸馏。

### Stage 3：轻量 GPT 或 KV-cache

两个选择：

1. 工程路线：保持权重不变，做 KV-cache。
2. 模型路线：训练 6-layer GPT student。

如果目标是大作业和 paper prototype，建议先走工程路线，因为风险更小，更容易解释“潜在完全流式架构”。

## 对“ODE=1 已经用默认原始模型”的理解

当前 ODE=1 不是蒸馏后的 1-step student，而是原始模型在推理时只跑一个 scheduler step。

这意味着：

- 模型训练时见过随机 timestep 的 noisy latent。
- 但它不是专门为 one-step endpoint 训练的。
- 所以 ODE=1 能跑，不代表它就是最优 one-step model。
- 如果 5-step 质量明显好于 1-step，可以蒸馏。
- 如果 1-step 质量已经可接受，优先优化 GPT 重算和 streaming state。

简单说：

```text
ODE=1 原始模型 = 少跑步数的 teacher
1-step distilled model = 专门学会一步复现多步 teacher 的 student
```

两者不是一回事。

## 最推荐的近期方案

我建议下一步不要立刻重做大型 FM 蒸馏，而是：

1. 固化 stream app 的 `stream_step(audio_chunk, state)` 接口。
2. 用这个接口保存 teacher rollout latent cache。
3. 训练 all-only + ODE=1 的 rollout-aware student。
4. 同时做 GPT incremental/KV-cache 的可行性验证。
5. 如果后续 1-step 质量仍不稳，再加 consistency/progressive FM distillation。

这条路线最符合当前 profiling：它把训练目标对齐真实流式推理，并且优先处理真正慢的 GPT 自回归重复计算。
