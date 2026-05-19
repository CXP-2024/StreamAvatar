# DyStream 实时流式推理优化方案

## 实测基准 (NVIDIA B20Z, 60s 音频, person2)

| 阶段 | 总耗时 | 每帧耗时 | 占实时预算% |
|------|--------|---------|------------|
| Audio encoding (speaker) | 0.26s | 0.2ms | 0% |
| Audio encoding (listener) | 0.03s | 0.0ms | 0% |
| **Motion generation (GPT+FM)** | **63.85s** | **42.6ms** | **106%** |
| **Video rendering** | **20.4s** | **13.6ms** | **34%** |
| **TOTAL** | **84.2s** | **56.2ms** | **140%** |

> 注：渲染 13.6ms/帧 是纯 GPU 推理时间（flow_estimator + face_generator），
> 不含模型加载(7.5s)和 moviepy 视频编码。main.py 日志中的 60+s render 包含了
> 模型加载 + 逐帧写文件 + moviepy mux 音频的开销。

| 指标 | 数值 |
|------|------|
| 目标帧率 | 25 fps (40ms/帧) |
| 当前每帧耗时 | 56.2ms (motion 42.6ms + render 13.6ms) |
| 当前实时比 | 1.40x（差 40%） |
| 差距 | 需 ~1.4x 加速 |

当前推理链路（每帧实测）：

```
音频切片(预计算) → GPT前向+扩散去噪(5步) → 渲染(flow+generator)
                   ────── 42.6ms ──────     ──── 13.6ms ────
                        (76%)                    (24%)
```

**关键发现：Motion Generation 占 76%，是主要瓶颈。渲染占 24%，也需要优化。**

---

## Phase 1：工程优化（无需训练）

### 1.1 GPT KV-Cache ⭐ 最高优先级

**问题**：stride=1，每生成1帧都对94帧历史重新计算全部 K/V/Attention。

**方案**：缓存历史帧的 K/V，新帧只计算增量。

**预期收益**：GPT 部分 ~5x 加速（95 token 全量 → 1 token 增量）

**实现要点**：
- 12 个 GPT Block 的 self-attention 需要维护 KV-Cache
- Cross-attention (anchor) 的 KV 是固定的，可一次性预计算
- RoPE 位置编码需要按实际 position 注入，而非序列位置
- 滑窗满 96 帧时，驱逐最旧的 KV 条目

**难度**：中。GPT Block 内混合了多种 attention，需要逐一适配。

---

### 1.2 torch.compile 渲染模型

**问题**：渲染模型（flow_estimator + face_generator）每帧约 30ms，未经图优化。

**方案**：对渲染模型使用 `torch.compile(mode="reduce-overhead")`。

**预期收益**：渲染 2-5x 加速（30ms → 6-15ms），参考 FLOAT 实测 9x 加速。

**实现要点**：
- 渲染模型结构固定，batch=1，非常适合 CUDA Graph
- 首次调用有 ~10s 编译开销，之后每帧复用
- 需检查模型中是否有动态 shape 或 CPU 操作（如 `torch.tensor(...).cuda()`）阻止 graph capture

**难度**：低。

---

### 1.3 流式音频编码

**问题**：当前整段音频一次性送入 Wav2Vec2，不兼容实时流入场景。

**方案**：分块 causal 音频编码。

**预期收益**：使系统能接受流式音频输入，不影响速度但解除"必须等音频全部到达"的约束。

**实现要点**：
- 代码已有 `make_attention_causal()` 补丁，attention 层可逐块处理
- 卷积 feature extractor 需要 overlap 缓冲（receptive field ~400 samples）
- 每到达 640 samples（1帧@16kHz/25fps）即可产出新特征
- 需要维护一个滑动缓冲区管理 context

**难度**：中。卷积边界处理需要小心。

---

### 1.4 Pipeline 并行（Motion Gen ‖ Rendering）

**问题**：motion gen 和 rendering 当前串行执行。

**方案**：用双 CUDA stream 重叠执行——生成帧 N+1 的 motion 时，并行渲染帧 N。

```
Stream A (motion): [gen 帧1] [gen 帧2] [gen 帧3] ...
Stream B (render):          [rnd 帧1] [rnd 帧2] ...
```

**预期收益**：~20-40%（取决于 GPU 利用率重叠程度）

**实现要点**：
- 两个 `torch.cuda.Stream()`，motion gen 在 stream A，render 在 stream B
- render 前需 `stream_b.wait_stream(stream_a)` 同步当前帧 latent
- 输出帧引入 1 帧延迟（40ms），对实时场景通常可接受

**难度**：低。但如果 GPU 已被单个操作占满，收益有限。

---

### Phase 1 综合预期（基于实测数据）

| 优化 | 加速效果 | 叠加后每帧 |
|------|---------|-----------|
| 原始 (实测) | — | 56.2ms (motion 42.6 + render 13.6) |
| +KV-Cache (GPT 部分) | motion ~2-3x → ~15-20ms | ~30-35ms |
| +torch.compile (render) | render ~2x → ~7ms | ~22-27ms |
| +Pipeline overlap | ~15% | ~20-24ms |

**Phase 1 结果预估：~25ms/帧（~40fps），突破实时线！✅**

> KV-Cache 是最关键一步：当前 42.6ms 中大量时间花在重复计算历史帧的 K/V。
> 实现 KV-Cache 后 motion gen 可降至 ~15-20ms，加上 render 13.6ms 即可接近 40ms。
> 再叠加 torch.compile 和 pipeline 即可稳定低于 40ms。

---

## Phase 1A：`stream_app.py` 快速验证入口（优先做）

目标不是先改原始 `app.py`，而是新增一个独立的轻量入口 `stream_app.py`，用于快速验证：

- 模型常驻后，第二次及后续推理能否低于音频时长；
- 10s 音频能否在 8s 内完成，即 RTF < 0.8；
- 哪些工程优化在 B200/B20Z 上已经足够，哪些仍需要 KV-cache。

### 设计原则

- **只保留 Custom Input**：上传一张参考图 + 一个单人音频/视频文件。
- **只支持 speaker-only**：先不做 listener audio / dyadic conversation。
- **参数极简**：保留 denoising steps、render batch size、FP16 开关；CFG 先使用单人默认值。
- **不改原始 `app.py`**：原 demo 保持可用，`stream_app.py` 专门用于速度验证和后续流式形态。
- **必须输出 profiling**：每次生成后显示各阶段耗时、FPS、RTF、cache hit/miss。

### 页面输入

| 输入 | 说明 |
|------|------|
| Reference Image | 单张人脸参考图 |
| Speaker Audio / Video | 单人音频，可上传 wav/mp3/mov/mp4 等可被 librosa/ffmpeg 读取的文件 |
| Denoising Steps | 默认 1，用于 fast mode；可切换 3/5 做质量对比 |
| Render Batch Size | 默认 32，根据显存可调 |
| FP16 Render | 默认开启 |

### 页面输出

| 输出 | 说明 |
|------|------|
| Generated Video | 生成后的 mp4 |
| Preprocessed Image | 裁剪 resize 后的参考图 |
| Masked Image | 调试用 mask 图 |
| Timing Report | 分阶段耗时、FPS、RTF、cache 命中情况 |

Timing report 至少包含：

```
model_load_or_cache
image_process
image_cache_hit
audio_load
motion_generation
render_batch
video_encode_mux
total
audio_duration
rtf = total / audio_duration
motion_fps
render_fps
end_to_end_fps
```

### `stream_app.py` 内部流程

```
上传参考图 + 上传音频
        |
        v
hash(image), hash(audio)
        |
        |-- image cache hit:
        |      复用 resized_pil / masked_pil / motion_latent / face_feat
        |
        |-- image cache miss:
        |      face detect -> crop/resize/mask -> motion_encoder -> face_encoder
        |
        v
读取音频 -> padding -> tensor
        |
        v
DyStream motion generator
        |
        v
batch renderer:
  flow_estimator + face_generator batch inference
        |
        v
ffmpeg pipe encode + audio mux
        |
        v
mp4 + timing report
```

### 快速收益优先级

#### P0：必须先做

1. **模型常驻复用**
   - 复用 `_dystream_model`、`_noise_scheduler`、`_vis_ctx`、`_face_detector`。
   - 这是当前 app 第二次明显变快的主要原因，`stream_app.py` 必须保留。

2. **batch renderer**
   - 替换当前逐帧渲染：`for frame -> flow_estimator -> face_generator`。
   - 改成按 batch 处理 motion latent。
   - 当前实测 357 帧 batch render + encode 约 2.26s，约 158 FPS。

3. **ffmpeg pipe / ffmpeg mux**
   - 避免 `imageio + moviepy` 慢路径。
   - 用 ffmpeg stdin 写 raw RGB frames，再用 ffmpeg 合音频。

4. **默认 fast preset**
   - 默认 `denoising_steps=1`。
   - 页面仍允许切到 3/5 做质量对比。

#### P1：很快能做，收益稳定

5. **reference image cache**
   - 按图片内容 hash 缓存：`resized_pil`、`masked_pil`、`motion_latent`、`face_feat`。
   - 同一个人物换音频时，跳过人脸检测、motion encoder、face encoder。

6. **audio decode cache**
   - 按音频文件 hash 缓存 librosa/ffmpeg 解码后的 waveform。
   - 同一段音频反复测试不同参数时，跳过音频解码。

7. **speaker-only 跳过 listener 输入**
   - 页面不提供 listener audio。
   - 后续可进一步改 motion model fast path：无 listener 时避免第二路 wav2vec 和 audio-other CFG 分支。

#### P2：需要改模型内部，暂缓

8. **speaker-only CFG branch pruning**
   - 若 `cfg_anchor=0`、无 listener、`cfg_audio_other=0`，理论上不必计算全部 5 路 CFG 分支。
   - 需要改 `one_clip_only_inference()` 内部 batch 拼接逻辑。

9. **KV-cache**
   - 长期正确方向，但实现成本比上面几项高。
   - 等 P0/P1 做完后，再判断是否仍需要。

10. **伪流式 chunk pipeline**
    - 输入仍然可以是完整音频，但内部按 1s/2s chunk 生成并渲染。
    - 用于展示流式形态：`motion chunk -> render chunk -> append/stream output`。

### 第一版验收标准

在 B200/B20Z 机器上，使用 warmed-up app：

| 测试 | 目标 |
|------|------|
| 6s 单人音频 | 总耗时 < 6s |
| 10s 单人音频 | 总耗时 < 8s |
| 30s 单人音频 | RTF < 0.8 |
| 输出报告 | 清楚显示 motion/render/encode 各阶段耗时 |

若 P0/P1 后仍无法达到 RTF < 0.8，再进入 KV-cache 或 CFG branch pruning。

---

## Phase 2：模型蒸馏（需要训练）

### 2.1 CFG 蒸馏 ⭐ 关键

**问题**：5 路 CFG 导致每帧实际跑 5 次前向。

**方案**：用教师模型的 5 路 CFG 组合输出作为 target，训练学生模型单次前向直接输出 guided result。

**预期收益**：5x 加速（所有模块统一）

**训练方法**：
1. 用当前模型生成大量 (输入条件, CFG 组合输出) 配对
2. 训练一个 single-forward 模型（同架构，去掉 null 条件分支）直接拟合 CFG 输出
3. 可选：用 progressive distillation 逐步从 5 路 → 3 路 → 1 路

**难度**：中高。需要训练数据和训练代码（当前未开源）。

---

### 2.2 扩散步数蒸馏

**问题**：每帧 5 步串行去噪。

**方案**：Consistency Distillation 或 Progressive Distillation，将 5 步降为 1 步。

**预期收益**：扩散部分 5x 加速

**训练方法**：
- 用 5 步教师生成 (噪声, 干净 latent) 配对
- 训练 1 步学生直接从噪声映射到干净 latent
- 6 层 DiffusionBlock 架构不变，只减步数

**难度**：中。Consistency Distillation 是成熟技术。

---

### 2.3 GPT 层数压缩

**问题**：12 层 Transformer 是否都必要？

**方案**：知识蒸馏，12 层 → 6 层。

**预期收益**：GPT 部分再 ~2x（叠加 KV-Cache 后效果更显著）

**训练方法**：
- 初始化：取教师的偶数层（layer 0,2,4,6,8,10）
- 用教师的中间隐藏状态做逐层蒸馏
- 微调若干 epoch

**难度**：中。

---

### Phase 2 综合预期

| 优化 | 加速效果 | 叠加后每帧 |
|------|---------|-----------|
| Phase 1 结果 | — | ~75ms |
| +CFG 蒸馏 (5→1) | 所有模块 5x | ~15ms |
| +扩散 1 步 | 扩散部分 5x | ~12ms |
| +GPT 6 层 | GPT 部分 2x | ~10ms |

**Phase 2 结果预估：~10-15ms/帧（66-100fps），远超实时。✅**

---

## Phase 3：系统级工程

### 3.1 CUDA Graph 固化

编译完成后，将整个推理路径固化为 CUDA Graph，消除 kernel launch 开销。

### 3.2 双缓冲输出

```
Buffer A: [正在写入当前帧]
Buffer B: [正在被播放器读取上一帧]
每帧交换 → 零拷贝输出
```

### 3.3 动态质量调节

- 监控每帧实际耗时
- 超时时：跳过扩散步 / 降低渲染分辨率 / 复用上一帧
- 保证输出永远不卡顿

### 3.4 compile 缓存持久化

- 首次编译产物保存到磁盘
- 后续启动直接加载，消除 10s warmup

---

## 总结

| 阶段 | 需要训练? | 预期帧耗时 | FPS | 实时? |
|------|----------|-----------|-----|-------|
| 当前 (实测) | — | 56.2ms | 17.8 | ❌ (差40%) |
| Phase 1 (KV-Cache + compile) | 否 | ~25ms | ~40 | ✅ (刚好) |
| Phase 2 (CFG蒸馏 + 步数蒸馏) | 是 | ~8-12ms | 80-125 | ✅✅ |
| Phase 3 (系统优化) | 否 | ~5-8ms | 125-200 | ✅✅✅ |

**最小可行路径**：Phase 1 (KV-Cache + torch.compile) 即可达到实时。

---

## 前置依赖

| 需求 | 状态 |
|------|------|
| DyStream 训练代码 | ❌ 未开源（Phase 2 必需） |
| 训练数据 (HDTF + RealTalk) | ✅ 公开可获取 |
| 推理代码 | ✅ 已有 |
| GPU (A100/B20Z) | ✅ 可用 |
| Phase 1 改造 | ✅ 可立即开始 |
