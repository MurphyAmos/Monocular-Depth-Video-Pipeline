# import open3d as o3d

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
    vid = cv2.VideoCapture("{Input Video File Name Here}")
# Get resolution using properties
    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
    width = (int((width)*.75))

    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
    height = (int((height)*.75))
    fps = int(vid.get(cv2.CAP_PROP_FPS))
    frame_list,depth_map_list = [], []
   

    batch_count, count, success =0, 0, True
    while success:
        success, image = vid.read() # Read frame
        if success:
            #resize images to half width
            image = cv2.resize(image,(width, height), interpolation=cv2.INTER_AREA)
            
            # Convert to PIL Image to send to predictions 
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            frame_list.append(pil_image)

            #call predictions upon PIL image 
            predictions = pipe(pil_image)
            #update the count of predictions and print it out/ current frame count
            count+=1
            print(count)
            depth_map = predictions["depth"]
            ##make a color array for depthmap
            depth_map = np.array(depth_map) 
            depth_map_color = cv2.cvtColor(depth_map, cv2.COLOR_GRAY2BGR)
            #append depthmap image to list
            depth_map_list.append(depth_map_color)
            
    video_name = 'depthMapRender.mp4'
    video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), int(fps), (width, height))
    #loop through list of images of depthmaps to video
    for depth_map in depth_map_list:
        video.write(depth_map)
    vid.release()
    video.release()
    #this is just incase :)
    cv2.destroyAllWindows()
    print("Video generated successfully!")
frame_to_depthMap()
