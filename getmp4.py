import numpy as np
import cv2
import os

fps = 60
frameSize = (500, 500)
# .avi -> DIVX
# .mp4 -> mp4v
out = cv2.VideoWriter('output_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, frameSize)

for i in range(0, 255):
    # Image array
    img = np.ones((500, 500, 3), dtype=np.uint8) * i
    out.write(img)

out.release()
