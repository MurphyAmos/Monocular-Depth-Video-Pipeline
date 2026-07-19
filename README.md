# Monocular Depth Video Pipeline

Converts standard RGB video into a rendered depth-map video, using monocular depth estimation applied frame by frame.

## What it does

Video is just a sequence of images, so this project takes that idea literally. Each frame of an input video is decoded, resized, and passed through a monocular depth estimation model. The resulting per-frame depth predictions are converted into images and re-encoded into a new video, producing a continuous depth-map rendering of the original footage.

## How it works

1. **Frame extraction.** OpenCV (`cv2.VideoCapture`) reads the source video frame by frame and resizes each frame (75% of original resolution) to reduce inference cost.
2. **Depth inference.** Each frame is converted to a PIL image and passed through a HuggingFace `depth-estimation` pipeline running [Depth-Anything V2 (Small)](https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf), producing a per-frame depth map.
3. **Depth map conversion.** Each predicted depth map is converted to a NumPy array and re-colored (`cv2.COLOR_GRAY2BGR`) so it can be written as a standard video frame.
4. **Video encoding.** Once every frame has been processed, the complete list of depth maps is encoded into an output `.mp4` using `cv2.VideoWriter`, matching the original video's FPS.

Processing is sequential and frame by frame, with no batching, so there's no discontinuity introduced at batch boundaries. Depth is still estimated independently per frame.

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
* OpenCV for video I/O and frame decoding/encoding
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

Update the 'vid' variable in `frame_to_depthMap()` (line 19) to point to your source file, then run:

```bash
python main.py
```

Output is written to `depthMapRender.mp4` in the working directory.

## Known limitations & Next Fixes
* **Memory Scaling With Video Resolution.** As the video resolution gets greater/higher in quality, the time it takes to run a depth-estimation on each frame increases. Possible fix would be down scaling all resolutions to a set "prediction-ready" resolution.
* **Frame-to-frame flicker.** Each frame's depth is estimated independently, with no temporal consistency between frames. This is a known limitation of naive per-frame monocular depth estimation on video. Individual frames are accurate, but the sequence can flicker slightly.
* **No camera intrinsics or 3D reconstruction yet.** This pipeline stops at 2D depth-map video generation. Extending frame-wise depth into a registered 3D point cloud or mesh is a natural next step and something I'm actively exploring.
* **No UI.** Currently, there is no UI. That's ok for a quick demo, but not only will it make the video generation process more tedious, but it will also increase the likelihood of in-code mistakes.
* ~~**Memory scales with video length.** All processed frames are held in memory as a list before the output video is encoded, instead of being written incrementally. That's fine for short clips but won't scale well to long videos.~~
##  Fixed & Updates
  
* **Memory Scaling with Length.** Video encoding now happens on a per-frame basis instead of buffering all frames in memory. This keeps memory usage flat regardless of video length, allowing the pipeline to scale to much longer videos without running out of memory.

* **Frame Skipping.** Implemented frame skipping every nth frame for faster processing at the cost of stream and video frame rate. This change increased overall execution speed by ~72%, freeing up memory usage. 

* **Live Video Streaming.** Depth Map Generation is now fed to a live video feed as an output. This will allow users to see the Depth Map Rendering process in real-time, instead of in the CLI. 
## Motivation

This started as an extension of a separate point-cloud and 3D reconstruction project. After working with depth maps on static images, the natural question was whether the same approach could be applied across an entire video, frame by frame, instead of just one frame at a time.
