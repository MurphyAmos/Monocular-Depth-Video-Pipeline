
import os
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
def frame_to_depthMap():
    vid = cv2.VideoCapture("{ORIGINAL_VIDEO}")
    # Get resolution using properties
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    width = (int((width)*.75))

    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    height = (int((height)*.75))
    fps = int(vid.get(cv2.CAP_PROP_FPS))
    
    video_name = '{OUTPUT_VIDEO}'
    video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), int(fps), (width, height))

    count, success = 0, True
    while success:
        success, image = vid.read() # Read frame
        if success:
            #resize images to half width
            image = cv2.resize(image,(width, height), interpolation=cv2.INTER_AREA)
            
            # Convert to PIL Image to send to predictions 
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

            #make predictions on it
            
            predictions = pipe(pil_image)
            count+=1
            print(count)
            depth_map = predictions["depth"]
            ##make a color array for depthmap
            depth_map = np.array(depth_map) 
            depth_map_color = cv2.cvtColor(depth_map, cv2.COLOR_GRAY2BGR)
            video.write(depth_map_color)

    #close all windows
    vid.release()
    video.release()
    cv2.destroyAllWindows()
    print("Video generated successfully!")
frame_to_depthMap()

