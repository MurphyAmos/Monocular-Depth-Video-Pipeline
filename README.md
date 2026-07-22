# Monocular Depth Video Pipeline

Converts a standard RGB live webcam feed, into a rendered depth-map video, using monocular depth estimation applied frame by frame.

## What it does

Video is just a sequence of images, so this project takes that idea literally. Each frame of an input is decoded, resized, and passed through a monocular depth estimation model. The resulting per-frame depth predictions are converted into images and re-encoded into a new video, producing a continuous depth-map rendering of the original footage.

## How it works

1. **Frame capture.** OpenCV (`cv2.VideoCapture`) reads frames from a live webcam feed, and resizes each frame to a fixed-target resolution to reduce inference cost.
2. **Depth inference.** Every nth frame is converted to a PIL image and passed through a HuggingFace `depth-estimation` pipeline running [Depth-Anything V2 (Small)](https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf), producing a per-frame depth map.
3. **Depth map conversion.** Each predicted depth map is converted to a NumPy array and re-colored (`cv2.COLOR_GRAY2BGR`) so it can be written as a standard video frame.
4. **Video encoding.** Depth maps are written to the output video incrementally, frame by frame, as they're generated, at an output framerate scaled to match actual capture rate.

Processing is sequential and frame by frame, with no batching, so there's no discontinuity introduced at batch boundaries. Depth is still estimated independently per frame.

Note: this is near-real-time when running off a live camera, not hard real-time — processing is throughput-bound by inference speed, so if inference is slower than the camera's native capture rate, the live preview will lag slightly behind what's currently in frame.

## Demo

<table>
  <tr>
    <td><img src="demos/original.gif" width="400"/></td>
    <td><img src="demos/depth.gif" width="400"/></td>
  </tr>
  <tr>
    <td align="center">Original</td>
    <td align="center">Depth-Map Output</td>
  </tr>
</table>

## Tech stack

* Python
* OpenCV for video I/O, frame decoding/encoding, and live camera capture
* HuggingFace Transformers for the model pipeline
* HuggingFace Accelerate for device selection (CPU/GPU)
* Depth-Anything V2 for monocular depth estimation
* Pillow and NumPy for image conversion

## Setup

Clone the repository:

```bash
git clone https://github.com/MurphyAmos/Monocular-Depth-Video-Pipeline.git
cd Monocular-Depth-Video-Pipeline
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Set your Hugging Face token:

```bash
export HF_TOKEN="your_token_here"      # macOS/Linux
setx HF_TOKEN "your_token_here"        # Windows
```
```python
import os
os.environ["HF_TOKEN"] = os.environ.get("HF_TOKEN")
```

## Usage
```bash
python main.py
```

Output is written to `LinkedIn-Test-2.mp4` in the working directory. Set the `preview` flag to `True` to also view the depth map live as it's generated (off by default for performance).

## Known limitations & Next Fixes
* **Frame-to-frame flicker.** Each frame's depth is estimated independently, with no temporal consistency between frames. This is a known limitation of naive per-frame monocular depth estimation on video. Individual frames are accurate, but the sequence can flicker slightly.
* **No camera intrinsics or 3D reconstruction yet.** This pipeline stops at 2D depth-map video generation. Extending frame-wise depth into a registered 3D point cloud or mesh is a natural next step and something I'm actively exploring.
* **No UI.** Currently, there is no UI. That's ok for a quick demo, but not only will it make the video generation process more tedious, but it will also increase the likelihood of in-code mistakes.
* ~~**Memory Scaling With Video Resolution.**~~
* ~~**Memory scales with video length.**~~

## Fixed & Updates
* **Live Camera Input.** The pipeline now accepts a live webcam feed instead of a video file, running near-real-time depth estimation frame by frame, throughput-bound by inference speed.
* **Preview On By Default.** Live preview via cv2.imshow is available through the preview flag, currently enabled by default so you can see the depth map as it's generated, keeping off runs faster.
* **Memory Scaling With Video Resolution.** Videos now downscale to a fixed target resolution instead of a relative percentage, keeping frame size (and memory load) consistent regardless of source resolution.
* **Memory Scaling with Length.** Video encoding now happens on a per-frame basis instead of buffering all frames in memory. This keeps memory usage flat regardless of video length, allowing the pipeline to scale to much longer videos without running out of memory.
* **Frame Skipping.** Implemented frame skipping every nth frame for faster processing at the cost of stream and video frame rate. This change increased overall execution speed by up to ~72%, freeing up memory usage.

## Motivation

This started as an extension of a separate point-cloud and 3D reconstruction project. After working with depth maps on static images, the natural question was whether the same approach could be applied across an entire video, frame by frame, instead of just one frame at a time.
