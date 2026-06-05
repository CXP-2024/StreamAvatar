# DyStream Distill 机制说明

## 目标

当前 DyStream motion generator 推理每帧大致是：

```
audio + history motion + anchor
        |
        v
GPT / AR backbone -> gpt_output
        |
        v
DiffusionHead 运行 N 个 ODE steps
        |
        v
motion latent
```

默认配置里 `denoising_steps=5`。蒸馏的目标是让 `DiffusionHead` 学会用 `1 step` 接近原始 teacher `5 steps` 的结果，从而减少每帧 diffusion head 的调用次数。

## 训练了什么

当前 distill 只训练两个组件：

```
model.diffusion_head
model.time_embed
```

冻结的组件：

```
wav2vec audio encoders
audio projection / fusion layers
GPT / AR backbone blocks
anchor / face embedding layers
renderer / visualization model
```

也就是说，蒸馏不是重新训练整个 DyStream，也不会改变 renderer。它只替换 motion generator 里负责 flow / denoising 的最后一段。

## Loss 是什么

训练时先用原始 teacher 生成 target：

```
noise + gpt_output branches
        |
        v
teacher DiffusionHead, 5-step CFG-combined denoise
        |
        v
teacher_target
```

再让 student 一步预测：

```
same noise + same gpt_output branches
        |
        v
student DiffusionHead, 1-step CFG-combined prediction
        |
        v
student_pred
```

loss 是：

```
MSE(student_pred, teacher_target)
```

当前实现已经按真实 inference 的 5 路 CFG 分支组合来生成 teacher target，而不是只蒸馏单路 diffusion output。

## 推理时如何替换

训练输出是一个轻量 checkpoint，例如：

```
outputs/distill/distilled_best.pt
outputs/distill/distilled_final.pt
```

里面只保存：

```python
{
    "diffusion_head": ...,
    "time_embed": ...,
    "step": ...,
    "loss": ...,
}
```

推理替换方式是：先正常加载完整 DyStream teacher checkpoint，然后把这两个子模块覆盖掉：

```python
distilled = torch.load("outputs/distill/distilled_best.pt", map_location=device)
model.diffusion_head.load_state_dict(distilled["diffusion_head"])
model.time_embed.load_state_dict(distilled["time_embed"])
```

替换后推理必须使用：

```python
num_inference_steps=1
```

因此当前不是“加载一个完整的新模型”，而是：

```
完整原始模型
    + 替换 diffusion_head
    + 替换 time_embed
    + denoising_steps=1
```

## 哪个接口开始替换

当前独立验证入口是：

```
verify_distill.py
```

它会分别加载：

1. 原始 teacher，跑 `5-step baseline`
2. 原始 teacher，跑 `1-step no distill`
3. 替换了 distilled diffusion head 的 student，跑 `1-step distilled`

命令示例：

```bash
.venv/bin/python verify_distill.py \
  --distilled outputs/distill/distilled_best.pt \
  --audio wav_files/woc.wav \
  --ref-image img_files/person1.png \
  --ref-npz img_files/person1.npz \
  --render
```

后续要接入当前 `app.py` 时，也是在模型加载完成后加入同样的覆盖逻辑。

## 当前 distill 不会减少参数量

这一步 distill 不改变模型结构：

```
DiffusionHead 参数量不变
GPT 参数量不变
Wav2Vec 参数量不变
Renderer 参数量不变
```

变快的原因是调用次数减少：

```
原始：每帧 diffusion_head forward 5 次
蒸馏后：每帧 diffusion_head forward 1 次
```

## 训练入口

训练使用：

```bash
.venv/bin/python train_distill.py \
  --config configs/distill/step_distill.yaml
```

短测试：

```bash
.venv/bin/python train_distill.py \
  --config configs/distill/step_distill.yaml \
  --override data.max_clips=10 training.batch_size=1 training.max_steps=100 training.num_workers=0
```

## 后续加速路线

### 1. 接入当前 app

在当前 `app.py` 中增加参数：

```
distilled_checkpoint: optional path
```

如果传入 distill checkpoint：

```
load full model
replace diffusion_head / time_embed
force denoising_steps = 1
```

### 2. Batch renderer + ffmpeg pipe

蒸馏只加速 motion generator。端到端速度还需要减少渲染和封装开销：

```
逐帧 renderer -> batch renderer
moviepy mux -> ffmpeg pipe / ffmpeg mux
```

### 3. CFG branch pruning

当前即使是 1-step，仍然保留 5 路 CFG branch：

```
uncond
anchor
audio
audio_other
all
```

speaker-only 场景下，如果：

```
no listener audio
cfg_anchor = 0
cfg_audio_other = 0
```

理论上可以裁掉不需要的分支，减少 batch 维度上的冗余计算。

### 4. CFG distillation

下一步可以让 student 直接预测 CFG-combined output：

```
5 CFG branches -> 1 guided branch
```

这会进一步减少 diffusion head 的 batch 维度计算，但需要改推理接口和训练目标。

### 5. KV-cache / chunk pipeline

长期流式形态仍然需要：

```
GPT KV-cache
audio feature cache
previous motion cache
chunk-level motion generation
chunk-level render / output
```

这部分主要降低 AR backbone 重复计算，并让系统从“整段离线生成”逐步变成“边来音频边出 motion/video chunk”。
