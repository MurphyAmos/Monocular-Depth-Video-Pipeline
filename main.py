
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
def frame_to_depthMap():
    vid = cv2.VideoCapture("vlipsy-michael-jackson-michael-jackson-moonwalk-nnsXoFYU.mp4")
    
    fc = 10
    fps = vid.get(cv2.CAP_PROP_FPS) / fc

    #get resolution    
    video_time = vid.get(cv2.CAP_PROP_FRAME_COUNT)/vid.get(cv2.CAP_PROP_FPS)
    src_width = vid.get(cv2.CAP_PROP_FRAME_WIDTH)
    src_height = vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
    
    #force scale each video...
    target_max = 120
    ##if width is bigger go on width else go on height for verticle
    if src_width >= src_height:
        scale = target_max / src_width
    else:
        scale = target_max / src_height

    width = int(src_width * scale)
    height = int(src_height * scale)
    video_name = 'depthMapRender.mp4'
    video = cv2.VideoWriter(video_name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

    start_time = time.perf_counter()    
    count, success, preview = 0, True, False
    while success:
        success, image = vid.read() # Read frame
        if success:
            count+=1
            if(count % fc == 0):
                #resize, convnert PIL, predict PIL
                image = cv2.resize(image,(width, height), interpolation=cv2.INTER_NEAREST)
                
                # Convert to PIL Image to send to predictions 
                predictions = pipe(Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)))
                depth_map = predictions["depth"]

                ##make a color array for writing depth_map to video
                depth_map = np.array(depth_map) 
                depth_map_color = cv2.cvtColor(depth_map, cv2.COLOR_GRAY2BGR)
                video.write(depth_map_color)
                if preview:
                    # Display the resulting frame in a window named 'Live Stream'
                    cv2.imshow('Depth Feed',depth_map_color)
                    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
                        break
    #when no longer successfully read the video,print out execution time
    end_time = time.perf_counter()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time:.4f} seconds")
    print(f"Original Vido Time: {video_time:.4f} seconds")

    #when no longer successfully read the video, release all resources
    vid.release()
    video.release()
    cv2.destroyAllWindows()

    #check final video length    
    check = cv2.VideoCapture(video_name)
    out_duration = check.get(cv2.CAP_PROP_FRAME_COUNT) / check.get(cv2.CAP_PROP_FPS)
    print("Output video duration: ", out_duration, " seconds")
    check.release()

    print("Video generated successfully!")
frame_to_depthMap()
