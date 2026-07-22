
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
#get user camera 
camera = cv2.VideoCapture(0)
def frame_to_depthMap():
    fc = 5
    #return frame and framecount
    source_fps = camera.get(cv2.CAP_PROP_FPS)/fc

    #get resolution    
    src_width = camera.get(cv2.CAP_PROP_FRAME_WIDTH)
    src_height = camera.get(cv2.CAP_PROP_FRAME_HEIGHT)

    #force scale each video...
    target_max = 480  
    ##if width is bigger go on width else go on height for verticle
    if src_width >= src_height:
        scale = target_max / src_width
    else:
        scale = target_max / src_height
    width = int(src_width * scale)
    height = int(src_height * scale)
    #set video parameters 
    video_name = 'LinkedIn-Test-2.mp4'
    video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), source_fps, (width, height))
    count, success, preview = 0, True,True
    while success:
        success, image = camera.read() # Read frame
        if success:
            count+=1
            #take in every 5 frame
            if(count % fc == 0):
                #resize, convnert PIL, predict PIL
                image = cv2.resize(image,(width, height), interpolation=cv2.INTER_NEAREST)
                
                # Convert to PIL Image to send to predictions 
                predictions = pipe(Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)))
                depth_map = predictions["depth"]

                ##make a color array for writing depth_map to video
                depth_map = np.array(depth_map) 
                depth_map_color = cv2.cvtColor(depth_map, cv2.COLOR_GRAY2BGR)
                #write video as frames are processed 
                video.write(depth_map_color)
                if preview:
                    # Display the resulting frame in a window named 'Live Stream'
                    cv2.imshow('Depth Feed',depth_map_color)
                    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
                        break
    #when no longer successfully read the camera, release all resources
    video.release()
    cv2.destroyAllWindows()
    print("Video generated successfully!")
frame_to_depthMap()
