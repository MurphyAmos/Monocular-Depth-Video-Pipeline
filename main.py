
import os, time
import numpy as np
from transformers import pipeline
from accelerate import Accelerator
import cv2
from PIL import Image

device = Accelerator().device
os.environ["HF_TOKEN"] = os.environ.get("HF_TOKEN")
pipe = pipeline(
    "depth-estimation",
    device = device,
    model="depth-anything/Depth-Anything-V2-Small-hf",
)
camera = cv2.VideoCapture(0)
fc = 8
source_fps = camera.get(cv2.CAP_PROP_FPS)/fc
#get resolution    
src_width = camera.get(cv2.CAP_PROP_FRAME_WIDTH)
src_height = camera.get(cv2.CAP_PROP_FRAME_HEIGHT)

#force scale each video...
target_max = 270
##if width is bigger go on width else go on height for verticle
if src_width >= src_height:
    scale = target_max / src_width
else:
    scale = target_max / src_height
width = int(src_width * scale)
height = int(src_height * scale)

def frame_to_depthMap():
    video_name = 'LinkedIn-Test-2.mp4'
    video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), source_fps, (width, height))
    count, success, preview = 0, True,True
    while success:
        count+=1
        if count % fc != 0:
            success = camera.grab() # Fetches frame from buffer but DOES NOT decode it
            if not success:
                break
            continue # Instantly skip to the next loop iteration
        success, image = camera.read() # Read frame
        if success:
            #resize, convnert PIL, predict PIL
            image = cv2.resize(image,(width, height), interpolation=cv2.INTER_NEAREST)
            
            # Convert to PIL Image to send to predictions 
            predictions = pipe(Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)))
            depth_map = predictions["depth"]

            ##make a color array for writing depth_map to video
            depth_map = np.array(depth_map) 
            depth_heatmap = cv2.applyColorMap(depth_map, cv2.COLORMAP_JET)
            video.write(depth_heatmap)
            if preview:
                combined = np.hstack((depth_heatmap, image))
                # Display the resulting frame in a window named 'Live Stream'
                cv2.imshow('Depth Feed',combined)
                if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
                    break
    #when no longer successfully read the video, release all resources
    video.release()
    cv2.destroyAllWindows()
    print("Video generated successfully!")
frame_to_depthMap()
