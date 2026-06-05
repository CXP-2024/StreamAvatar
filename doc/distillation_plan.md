# DyStream 蒸馏加速方案

## 你的数据集：LRS3 trainval

| 属性 | 数值 |
|------|------|
| 路径 | `/mnt/pfs/group-jt/zihan.guo/droid/LRS3/lrs3/trainval/` |
| 视频数 | 31,982 clips |
| 说话人数 | 4,004 |
| 平均时长 | ~3.4s/clip |
| 总时长 | ~30 小时 |
| 分辨率 | 224×224, 25fps |
| 音频 | 16kHz mono (AAC in MP4) |
| 语种 | 英语 (TED talks) |
| 标注 | 文本转写 + 置信度分数 |

---

## 当前瓶颈回顾

```
每帧 56.2ms = Motion Gen 42.6ms (76%) + Render 13.6ms (24%)
```

Motion Gen 的 42.6ms 拆解（推算）：
- GPT 前向 (12层, 95 token, ×5 CFG batch): ~8ms × 5 = ~8ms (batched)
- 扩散去噪 (6层 DiffHead, 5步, ×5 CFG batch): ~7ms × 5步 = ~35ms
- **扩散去噪是 motion gen 的主要成本**

---

## 蒸馏目标与预期收益

| 蒸馏目标 | 当前 → 目标 | 预期加速 | 对总时间影响 | 难度 |
|----------|------------|---------|-------------|------|
| **扩散步数 5→1** | 5步×5CFG=25次 → 1步×5CFG=5次 | motion ~5x | **42.6→~10ms** | ⭐ 低 |
| CFG 5→1 路 | 5路 → 1路 | 再 ~5x | 10→~3ms | 中 |
| GPT 12→6 层 | 12层 → 6层 | GPT ~2x | 微小（GPT 非瓶颈）| 中 |

**推荐最小改动方案：只做扩散步数蒸馏 (5→1)**

理由：
1. 收益最大（motion gen 从 42.6ms 降到 ~10ms，总体 56→~24ms，直接实时）
2. 实现最简单（不改模型架构，只改训练 loss）
3. 不需要 LRS3 视频帧（可以纯用音频做 output distillation）
4. 30 小时数据绰绰有余

---

## 推荐方案：扩散步数蒸馏 (5→1)

### 原理

```
教师 (5步):  noise → step1 → step2 → step3 → step4 → clean_latent
学生 (1步):  noise ─────────────────────────────────→ clean_latent
```

学生模型架构不变（同样的 6 层 DiffusionBlock），只是学习一步直达。

### 训练方法：Progressive Distillation

```
Round 1: 5步教师 → 3步学生  (步数减半取整)
Round 2: 3步教师 → 1步学生  (最终目标)
```

或直接用 **Consistency Distillation**：
- 从教师的 ODE 轨迹上采样 (t, x_t) 对
- 训练学生满足自一致性：f(x_t, t) = f(x_{t'}, t') 对同一轨迹上的任意两点

### 训练数据生成

**方案 A（推荐）：纯音频 output distillation — 不需要预处理视频**

```python
# 用教师模型跑推理，收集 (输入, 输出) 对
for audio_clip in LRS3_trainval_audios:
    noise = torch.randn(...)  # 随机噪声
    teacher_output = teacher.inference(audio, steps=5)  # 5步教师输出

    # 保存训练对
    save(noise, teacher_output, audio_features, timestep)
```

优点：
- **零预处理**：直接从 LRS3 提取音频即可
- 不依赖视频分辨率（224×224 的限制无影响）
- 教师输出就是"ground truth"

缺点：
- 学生上限是教师质量（不能超越教师）

**方案 B：使用视频帧提取 motion latent 做正则化**

```python
# 需要预处理：视频帧 → motion latent
for video_clip in LRS3_trainval:
    frames = extract_frames(video_clip)  # 224×224
    frames_resized = resize(frames, 512×512)  # 上采样
    motion_latents = motion_encoder(frames_resized)  # 每帧→512d

    # 保存 motion latent 序列 + 对应音频
    save(motion_latents, audio)
```

优点：
- 有真实 motion 做正则化，防止退化
- 可同时训练 motion gen 主模型

缺点：
- 需要预处理全部 31,982 个视频
- 224→512 上采样有质量损失
- 预处理需要 GPU 时间（约 4-6 小时）

---

## 数据预处理需求

### 方案 A（推荐，最小改动）

| 步骤 | 是否需要 | 说明 |
|------|---------|------|
| 提取音频 | ✅ | `ffmpeg -i video.mp4 -vn -ar 16000 -ac 1 audio.wav` |
| 提取视频帧 | ❌ | 不需要 |
| 上采样到 512×512 | ❌ | 不需要 |
| 提取 motion latent | ❌ | 不需要 |
| 运行教师模型 | ✅ | 用教师生成 (noise, output) 对 |

**预处理工作量**：
- 提取音频：~1 小时（31,982 个 ffmpeg 调用，可并行）
- 运行教师模型：~30 小时音频 ÷ 实时比 1.4x ≈ **~42 小时 GPU 时间**（可用多卡加速）
- 总预处理：**1-2 天**（单卡），几小时（多卡）

### 方案 B（如果你也想微调主模型）

| 步骤 | 是否需要 | 说明 |
|------|---------|------|
| 提取音频 | ✅ | 同上 |
| 提取视频帧 | ✅ | 25fps, resize 到 512×512 |
| 提取 motion latent | ✅ | 用 DyStream 的 motion_encoder 逐帧推理 |
| 生成 metadata JSON | ✅ | DyStream 训练格式 |

**额外预处理工作量**：
- 帧提取 + resize：~2-3 小时
- Motion latent 提取：31,982 clips × 85 frames avg × 13.6ms ≈ **~10 小时 GPU**
- 总额外：**~12 小时**

---

## 哪些组件适合训练？

### ✅ 扩散头 (DiffusionBlock × 6) — 最推荐

**为什么适合**：
- 参数量小（6 层 MLP，相对于整个 7.2GB 模型只是一小部分）
- 训练目标明确：从噪声直接预测干净 latent
- 不改变 GPT backbone，保留其已学到的时序建模能力
- 收敛快（通常 5k-20k steps 即可）

**训练配置建议**：
```yaml
# 只训练 diffusion_head，冻结其余
trainable: model.diffusion_head  # 6层 DiffusionBlock
frozen: [model.gpt_blocks, model.audio_encoder_face, model.audio_encoder_face_other, ...]
lr: 1e-4
batch_size: 32
steps: 10000-20000
loss: MSE(student_1step_output, teacher_5step_output)
```

**30 小时数据是否足够**：✅ 绰绰有余。
- 30h × 25fps = ~270 万帧
- 每帧可生成多个 (noise, output) 对（不同随机种子）
- 扩散头参数量 ~10M，10k steps × batch32 = 32 万样本即可收敛

---

### ⚠️ GPT Backbone (12 层 Transformer) — 可选但非必须

**为什么排优先级较低**：
- GPT 不是当前瓶颈（实现 KV-Cache 后更不是）
- 压缩 GPT 容易损伤时序建模质量（表情连贯性）
- 训练成本高（需要端到端训练）

**如果要做**：
- 用教师 12 层的中间隐藏状态监督学生 6 层
- 初始化：取教师偶数层 (0,2,4,6,8,10)
- 需要更多数据和更长训练

---

### ❌ Audio Encoder (Wav2Vec2) — 不建议训练

**为什么不动**：
- 音频编码只跑一次，不是 per-frame 瓶颈
- Wav2Vec2 预训练很充分，fine-tune 容易退化
- 冻结它可以保证音频特征质量

---

### ❌ Render Model (flow_estimator + face_generator) — 不建议蒸馏

**为什么不动**：
- 已经很快（13.6ms/帧）
- 用 torch.compile 就能进一步加速到 ~7ms
- 蒸馏渲染模型需要像素级监督数据，复杂度高

---

## 实施路线图

```
Week 1: 数据准备
├── 从 LRS3 trainval 提取全部音频 (31,982 clips)
├── 用教师模型生成 distillation 训练对
│   └── 输入: audio + random_noise + timestep
│   └── 输出: teacher_5step_clean_latent
└── 整理成训练 dataloader 格式

Week 2: 蒸馏训练
├── 冻结全部模型，只训练 diffusion_head
├── Loss = MSE(student_1step(noise, audio, t=1.0), teacher_output)
├── 训练 ~10k-20k steps (batch=32, 单卡 ~6-12 小时)
└── 验证：对比 1步 vs 5步 的 motion latent 质量

Week 3: 集成与验证
├── 将 distilled diffusion_head 替换原模型
├── 推理时设置 denoising_steps=1
├── 端到端测试 motion gen 速度
└── 视频质量主观评估 + 定量指标 (FID, lip-sync)
```

---

## 预期最终效果

| 配置 | Motion Gen | Render | 总计 | FPS | 实时? |
|------|-----------|--------|------|-----|-------|
| 当前 (5步, 5CFG) | 42.6ms | 13.6ms | 56.2ms | 17.8 | ❌ |
| **蒸馏后 (1步, 5CFG)** | **~10ms** | **13.6ms** | **~24ms** | **~42** | **✅** |
| 蒸馏 + KV-Cache | ~5ms | 13.6ms | ~19ms | ~53 | ✅✅ |
| 蒸馏 + KV-Cache + compile | ~5ms | ~7ms | ~12ms | ~83 | ✅✅✅ |

**仅做扩散步数蒸馏一项**，motion gen 从 42.6ms 降到 ~10ms，总体 24ms/帧，即可达到实时。
这是**最小改动、最大收益**的方案。

---

## 关键风险与缓解

| 风险 | 概率 | 缓解 |
|------|------|------|
| 1步生成质量下降 | 中 | 先试 2步折中；或用 consistency loss 替代纯 MSE |
| 224→512 上采样损伤 motion latent | 低（方案A不涉及）| 用方案 A，不依赖视频帧 |
| LRS3 英语数据 vs 中文推理场景 | 低 | 蒸馏的是扩散头的去噪能力，与语种无关 |
| 训练代码未开源 | 中 | 蒸馏只需训练 diffusion_head，逻辑简单可自行编写 |
| 30h 数据量不够 | 极低 | 扩散头参数 ~10M，30h 远超需求 |

---

## 总结

| 问题 | 答案 |
|------|------|
| 蒸馏能提升多少? | motion gen 约 **4-5x**（42.6ms→~10ms），总体 **2.3x**（56ms→24ms） |
| 训练哪个组件? | **只训练 DiffusionHead (6层)**，冻结其余全部 |
| 数据需要预处理吗? | **方案 A 几乎不需要** — 只提取音频 + 跑教师模型生成训练对 |
| 数据量够吗? | **绰绰有余** — 30h 对 ~10M 参数的扩散头来说远超需求 |
| 最小改动? | 不改架构，不改 GPT，不改 audio encoder，只替换 diffusion_head 权重 |
| 多久能做完? | 数据准备 1-2 天 + 训练 6-12 小时 + 验证 1 天 ≈ **一周内** |
