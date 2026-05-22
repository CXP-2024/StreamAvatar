# Rolling Stream App Plan

## 当前问题

当前 stream app 里的 `chunk_frames` 主要是调度参数。它把若干逐帧 AR step 打包返回，但还没有形成真正的跨请求 rolling state。早期的 frame chunk 播放版本也容易出现 chunk 间不连续，因为每个 chunk 可能重新初始化 motion/audio context。

目标是改成更接近真实流式：

```text
audio chunk arrives
  -> rolling audio window encode
  -> GPT/FM uses preserved past_motion
  -> output only the new chunk frames
  -> renderer renders new frames
  -> frontend appends frames at 25 fps
```

## 约束

DyStream 当前 GPT 的 native history 是：

```text
model.inpainting_length = cbh_window_length - 2 = 94 frames
```

也就是约：

```text
94 / 25 = 3.76s
```

所以第一版为了不改模型结构，GPT rolling history 仍用 94 frames。用户提到的 2s audio window 是合理目标，但如果直接把 audio/GPT history 都缩到 50 frames，需要额外改模型里的 `inpainting_length` 和 checkpoint 分布，风险更高。第一版先做 native-history rolling，保证连续性。

## 第一版后端行为

输入完整音频时，后端模拟流式：

```text
chunk_frames = 8
hop = 16000 / 25 = 640 samples

for generated_start in 0, 8, 16, ...:
    gen = min(8, remaining_frames)
    audio_window = padded_audio[generated_start : generated_start + 94 + gen + 1]
    audio_features = Wav2Vec2(audio_window)
    motion_chunk = one_clip_only_inference(
        precomputed_audio_features,
        past_motion,
        gen_frames=gen,
    )
    past_motion = concat(past_motion, motion_chunk)[-94:]
    append motion_chunk
```

这里有一个 `+1 frame` 的 audio lookahead，这是原 DyStream 推理代码里的对齐方式：

```text
n = gen_frames + inpainting_length + 1
audio_features = audio_features[:, 1:]
```

因此如果 `chunk_frames=8`，纯算法等待大约是：

```text
8 / 25 + 1 / 25 = 0.36s
```

再加上模型计算和渲染时间。

## 为什么这样能减少不连续

跨 chunk 保留：

- `past_motion`
- anchor latent
- model/renderer state

每个 chunk 不再从 reference latent 重新开始，而是使用上一 chunk 生成的最后 94 帧作为历史。这样 GPT 看到的是连续 motion history。

## 前端播放 25 fps 的方式

最终理想前端不是反复替换 `<video>`，而是维护一个 frame queue：

```text
producer: backend returns frames for each chunk
consumer: browser timer at 25 fps pops one frame and draw to canvas
```

播放策略：

- 先缓存 1-2 个 chunk，避免播放 underrun。
- 每 40ms 显示一帧。
- 如果后端慢，重复最后一帧或轻微等待。
- 如果后端快，queue 累积但按 25 fps 消费。

这样视觉上是连续帧流，不是一个个 mp4 文件刷新。

## 实现阶段

### Phase 1: 后端 rolling motion

在 `stream_app.py` 增加 rolling mode：

- 按 chunk 切音频。
- 每 chunk 只编码当前 rolling audio window。
- 调用 `one_clip_only_inference`。
- 保留 `past_motion`。
- 输出最终 mp4 和 chunk timing。

### Phase 2: 状态化接口

新增 session state：

```text
StreamState:
  past_motion
  generated_frames
  audio_tail
  reference image/latent
  timing history
```

每次浏览器上传 audio chunk 时只处理新增 chunk。

### Phase 3: canvas frame player

前端改成：

- 上传参考图后初始化 state。
- 麦克风每 `chunk_frames / 25` 秒发送 chunk。
- 后端返回 JPEG/PNG frame list 或 raw encoded frames。
- 浏览器 canvas 以 25 fps 消费 frame queue。

### Phase 4: 更短 context 实验

如果 native 94-frame history 延迟/计算仍太重，再试：

- audio context 50 frames。
- GPT history 50 frames。
- 或 block-wise student 直接按 K=2/8 输出。

