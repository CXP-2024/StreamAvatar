# DyStream Audio2FaceGPT 架构参考

## ⚠️ 重要发现：Config 与实际架构不一致

Config 文件 `custom_current.yaml` 中定义了以下参数，但 **实际模型并未使用它们**：

| Config 参数 | Config 值 | 实际值 (Python 默认) | 说明 |
|------------|-----------|---------------------|------|
| `layers` | 4 | **12** | GPT 层数 |
| `hidden_size` | 512 | **768** | Transformer hidden dim |
| `heads` | 8 | **12** | Attention heads |
| `ffn_dim` | 2048 | **3072** (768×4) | MLP hidden dim |
| `attention_head_dim` | 64 | **64** (768/12) | 恰好相同 |
| `pose_length` | 32 | — | 未被模型引用 |

**原因**：`Audio2FaceGPT(cfg)` 只接收 `cfg` 作为第一个参数，其余 `hidden_size`, `num_layers` 等走的是 Python 构造函数的默认值。Config 中的这些字段可能是早期版本遗留或用于其他模型变体。

---

## 实际被使用的 Config 参数

| Config 参数 | 值 | 用途 |
|------------|-----|------|
| `wav2vec_layer` | 8 | Wav2Vec2 只保留前 8 层（原 12 层） |
| `cbh_window_length` | 96 | 滑动窗口总长度 |
| `pose_fps` | 25 | 视频帧率，用于音频特征时间插值 |
| `audio_sr` | 16000 | 音频采样率 |
| `vae_codebook_size` | 512 | = motion latent 维度 (face_dim) |
| `prev_audio_frames` | 32 | 音频预填充帧数 |
| `denoising_steps` (validation) | 5 | 推理时扩散步数 |
| `cfg_audio` | 0.5 | 自身音频 CFG 强度 |
| `cfg_audio_other` | 0.5 | 对方音频 CFG 强度 |
| `cfg_anchor` | 0 | Anchor CFG（**当前为 0，等于禁用**）|
| `cfg_all` | 1.0 | 全条件 CFG 强度 |
| `cfg_fusion` | 2.0 | 在 inference() 中用作 `cfg_all` 的实际值 |

---

## 模型架构总览

```
Audio2FaceGPT
├── audio_encoder_face        : WrapedWav2Vec (Wav2Vec2-base, 截断到 8 层)
├── audio_encoder_face_other  : WrapedWav2Vec (同上，独立权重)
├── audio_proj                : Linear(768 → 768)
├── audio_other_proj          : Linear(768 → 768)
├── audio_audioother_fusion   : Linear(1536 → 768)
├── face_embed                : Linear(512 → 768)
├── anchor_embed              : Linear(512 → 768)
├── time_embed                : WanTimeEmbedding(dim=768)
├── blocks × 12               : Audio2FaceGPTBlock(hidden=768, heads=12, mlp=3072)
├── output_norm               : LayerNorm(768)
├── output_proj               : Linear(768 → 512)
└── diffusion_head            : DiffusionHead(face=512, hidden=768, blocks=6)
```

---

## 各模块详细结构

### WrapedWav2Vec (×2 实例)

```
WrapedWav2Vec
├── feature_extractor     : 7 层 CNN (16kHz → ~50fps, dim 512)
├── feature_projection    : Linear(512 → 768)
├── encoder               : 8 层 Wav2Vec2TransformerEncoder
│   └── 每层: SelfAttn(768, 12 heads) + MLP(768→3072→768)
└── 返回: {"high_level": encoder_output} shape [bs, T_audio, 768]
```

参数量：~94M（每个实例）

---

### Audio2FaceGPTBlock (×12)

```
Audio2FaceGPTBlock(hidden_size=768, num_heads=12, mlp_ratio=4.0)
├── norm1           : LayerNorm(768)
├── self_attn_rope  : RoPE Self-Attention (768, 12 heads, causal)
│
├── cross_linear_audio : Linear(768 → 768)  ← 音频条件注入（加法）
│
├── norm_anchor     : LayerNorm(768)
├── cross_attn_anchor : RoPE Cross-Attention (768, 12 heads) → 对 anchor 做交叉注意力
│
├── norm3           : LayerNorm(768)
├── self_attn_pos   : Sinusoidal PE Self-Attention (768, 12 heads, causal)
│
├── norm4           : LayerNorm(768)
└── mlp             : Linear(768→3072) + GELU + Linear(3072→768)
```

每个 block 的前向流：
```
x → LN → RoPE_SelfAttn(causal) → +residual
  → + audio_linear(audio_fea)    ← 简单线性加法注入
  → LN → CrossAttn(anchor)      → +residual
  → LN → Pos_SelfAttn(causal)   → +residual
  → LN → MLP                    → +residual
→ 输出 x
```

注：`norm2` 和 `cross_attn_rope` 定义了但 **forward 中未使用**（死代码）。

---

### DiffusionHead (6 个 DiffusionBlock)

```
DiffusionHead
├── noisy_proj     : Linear(512 → 768)     ← 噪声 latent 投影
├── gpt_proj       : Linear(512 → 768)     ← GPT 输出投影
├── blocks × 6     : DiffusionBlock(768)
├── output_norm    : LayerNorm(768)
└── output_proj    : Linear(768 → 512)     ← 输出 motion latent
```

每个 DiffusionBlock：
```
DiffusionBlock(hidden_size=768)
├── adaLN_modulation1 : Linear(768 → 768×3)  ← 由 time_embedding 调制
├── norm1             : LayerNorm(768)
├── mlp1              : Linear(768→3072) + GELU + Linear(3072→768)
│
├── adaLN_modulation2 : Linear(768 → 768×3)  ← 由 gpt_hidden 调制
├── norm2             : LayerNorm(768)
└── mlp2              : Linear(768→3072) + GELU + Linear(3072→768)
```

**没有注意力层** — 纯 MLP + Adaptive LayerNorm。

---

## 推理时 Shape 流

### 输入

```
audio_waveform:  [1, N_samples]     N_samples = 60s × 16000 = 960,000
motion_latent:   [1, 1, 512]        参考图的 motion latent
anchor_motion:   [1, 1, 512]        = motion_latent（参考帧）
```

### Stage 1: 音频编码（一次性）

```
audio_waveform [1, 960000]
    ↓ Wav2Vec2 feature_extractor (CNN, ~320x downsample)
    [1, ~3000, 512]
    ↓ feature_projection
    [1, ~3000, 768]
    ↓ encoder (8 transformer layers)
    [1, ~3000, 768]                         ← ~50fps 的音频特征
    ↓ F.interpolate(scale_factor=25/50)     ← 从 50fps 插值到 25fps
    [1, ~1500, 768]                         ← = total_frames (T)

同理 audio_other → [1, T, 768]
```

### Stage 2: 音频特征融合（一次性）

```
audio_proj(speaker_fea):        [1, T, 768]
audio_other_proj(other_fea):    [1, T, 768]
    ↓ cat dim=-1
    [1, T, 1536]
    ↓ audio_audioother_fusion
    [1, T, 768]                             ← fused_audio_fea
```

### Stage 3: 逐帧自回归 Motion Generation

对每一帧 t（共 T 帧，stride=1）：

```
window = 96 tokens = [94 历史帧 + 1 当前帧 + 1 偏移]

── 5 路 CFG 复制 ──
face_hidden:    [5, 95, 768]    ← 历史 face embedding + 当前位置
audio_hidden:   [5, 95, 768]    ← 5 种条件组合（见下方 CFG 表）
anchor_hidden:  [5, 1, 768]     ← 5 种条件组合

── 12 个 GPT Block ──
x:              [5, 95, 768]    ← 全序列处理，causal mask
    ↓ 12 × Audio2FaceGPTBlock
x:              [5, 95, 768]

── 取最后一个位置 ──
x[:, -1:]:      [5, 1, 768]
    ↓ output_norm + output_proj
gpt_output:     [5, 1, 512]

── 扩散去噪（5 步）──
noise:          [1, 1, 512]     ← 随机噪声（无 CFG batch）
for step in [1..5]:
    noise_5x:   [5, 1, 512]    ← 复制 5 份
    ↓ noisy_proj
    [5, 1, 768]
    ↓ gpt_proj(gpt_output)
    [5, 1, 768]
    ↓ 6 × DiffusionBlock (conditioned on time_emb + gpt_hidden)
    [5, 1, 768]
    ↓ output_proj
    velocity:   [5, 1, 512]
    ↓ CFG 组合
    guided_vel: [1, 1, 512]
    ↓ Euler step (flow matching scheduler)
    noise:      [1, 1, 512]     ← 更新后的 latent

── 输出 ──
denoised:       [1, 1, 512]     ← 当前帧的 motion latent
```

### Stage 4: 渲染（逐帧）

```
motion_latent[t]:  [1, 512]
ref_latent:        [1, 512]
    ↓ flow_estimator(ref_latent, driving_latent)
    flow_field (optical flow)
    ↓ face_generator(flow_field, face_features)
    output_frame:  [1, 3, 512, 512]
```

---

## 5 路 CFG 条件组合

| Branch | Self Audio | Other Audio | Anchor | 用途 |
|--------|-----------|-------------|--------|------|
| 0 | ✗ (零) | ✗ (零) | ✗ (零) | 完全无条件 |
| 1 | ✗ (零) | ✗ (零) | ✓ | 仅 anchor |
| 2 | ✓ | ✗ (零) | ✗ (零) | 仅自身音频 |
| 3 | ✗ (零) | ✓ | ✗ (零) | 仅对方音频 |
| 4 | ✓ | ✓ | ✓ | 全条件 |

**组合公式**：
```
output = uncond
       + cfg_audio       × (audio_only - uncond)       # 0.5
       + cfg_audio_other × (other_only - uncond)       # 0.5
       + cfg_anchor      × (anchor_only - uncond)      # 0 (禁用!)
       + cfg_all         × (full_cond - uncond)        # 1.0 (实际用 cfg_fusion=2.0)
```

**注意**：`cfg_anchor=0` 意味着 branch 1 的计算完全浪费！可安全移除为 4 路。

---

## 每帧计算量分析

以 window=95 tokens, batch=5 (CFG), hidden=768 为例：

| 操作 | 形状 | FLOPs 估算 |
|------|------|-----------|
| GPT Self-Attn (×12 层, ×2 次/层) | [5, 95, 768] | 12×2× (2×95²×768×5) ≈ 13.3G |
| GPT Cross-Attn (×12 层) | [5,95,768]×[5,1,768] | 12× (2×95×1×768×5) ≈ 0.09G |
| GPT MLP (×12 层) | [5, 95, 768→3072→768] | 12× (2×95×768×3072×5) ≈ 27.1G |
| GPT 小计 | | **~40G FLOPs** |
| Diffusion MLP (×6 block, ×5 步) | [5, 1, 768→3072→768] ×2 | 6×5× (2×2×768×3072×5) ≈ 0.57G |
| Diffusion 小计 | | **~0.6G FLOPs** |
| **合计** | | **~41G FLOPs** |

**关键洞察**：GPT 占 **98%** 的计算量，扩散头只占 2%！

但为什么实测中 motion gen=42.6ms 看起来扩散占比很高？

因为 **GPT 对 95 个 token 做的是全量注意力（无 KV-Cache）**，但由于 GPU 并行度高，95 token 的 transformer 跑得很快。而扩散头虽然 FLOPs 少，但有 **5 步串行依赖**（每步等上步结果），导致 GPU 利用率低。

---

## 参数量估算

| 模块 | 参数量 |
|------|--------|
| audio_encoder_face (WrapedWav2Vec, 8层) | ~94M |
| audio_encoder_face_other (同上) | ~94M |
| audio_proj + other_proj + fusion | ~2.4M |
| face_embed + anchor_embed | ~0.8M |
| time_embed (WanTimeEmbedding) | ~1.2M |
| GPT blocks × 12 | ~12 × 28M = **~336M** |
| output_norm + output_proj | ~0.4M |
| diffusion_head (6 DiffusionBlock) | ~**57M** |
| **总计** | **~586M** |

checkpoint 7.2GB 包含 EMA 副本 + optimizer states，实际模型 ~2.3GB (FP32)。

---

## 架构示意图

```
                    ┌─────────────────────────────────────────────┐
                    │           Audio2FaceGPT                       │
                    │                                               │
  speaker_audio ──→ │ WrapedWav2Vec(8L) → audio_proj ─┐            │
                    │                                  ├→ fusion ──→│──┐
  other_audio ───→ │ WrapedWav2Vec(8L) → other_proj ─┘            │  │
                    │                                               │  │
  ref_motion ────→ │ anchor_embed ──────────────────────────────→ │  │ anchor_hidden
                    │                                               │  │
  past_frames ───→ │ face_embed ────────────────────────────────→ │  │ face_hidden
                    │                                               │  │
                    │         ┌─── × 12 GPT Blocks ────┐           │  │
                    │         │ RoPE SelfAttn (causal)  │           │  │
                    │         │ + audio (linear add)    │◀──────────│──┘ audio_hidden
                    │         │ CrossAttn → anchor      │◀──────────│──── anchor_hidden
                    │         │ Pos SelfAttn (causal)   │           │
                    │         │ MLP (768→3072→768)      │           │
                    │         └────────────────────────┘           │
                    │                    ↓                          │
                    │         output_proj: 768 → 512                │
                    │                    ↓ gpt_output               │
                    │                                               │
  timestep ──────→ │ WanTimeEmbed ──→ time_embedding               │
                    │                                               │
  noise ─────────→ │      ┌── × 6 DiffusionBlock ──┐              │
                    │      │ AdaLN(time_emb) + MLP  │              │
                    │      │ AdaLN(gpt_out) + MLP   │              │
                    │      └────────────────────────┘              │
                    │                    ↓                          │
                    │         output_proj: 768 → 512                │
                    │                    ↓                          │
                    │            denoised_motion [1, 512]           │
                    └─────────────────────────────────────────────┘
```

---

## 对蒸馏计划的修正

之前文档中假设"扩散去噪是主要时间成本"，但从 FLOPs 分析看：

- **GPT 占 98% FLOPs**（40G vs 0.6G）
- 扩散头 FLOPs 很少，但有 5 步串行延迟

实测 42.6ms 中的真实分布（推测）：
- GPT 12层×5batch 全量前向: ~30ms（高并行度，GPU 利用率高）
- 扩散 5步串行: ~12ms（低并行度，GPU 等待多）

**修正后的蒸馏优先级**：
1. **步数蒸馏 5→1**：消除串行，节省 ~10ms → motion 降到 ~32ms
2. **KV-Cache**：避免 94 帧重计算，GPT 从 ~30ms 降到 ~3ms → motion 降到 ~5ms
3. **CFG 5→4 路**：移除无用的 anchor branch（cfg_anchor=0），免费 20% 加速

这三项组合后：motion gen 预计 **~4-5ms/帧**。
