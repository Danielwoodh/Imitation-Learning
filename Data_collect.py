import numpy as np
import os
import time
import cv2

# Initializing Camera_1
video_capture_0 = cv2.VideoCapture(2)
# Setting read resolution
video_capture_0.set(3, 48)
video_capture_0.set(4, 48)

# Setting capture resolution
cap_size0 = (int(video_capture_0.get(3)), int(video_capture_0.get(4)))

# Initializing Camera_2
video_capture_1 = cv2.VideoCapture(0)
# Setting read resolution
video_capture_1.set(3, 48)
video_capture_1.set(4, 48)

# Setting capture resolution
cap_size1 = (int(video_capture_1.get(3)), int(video_capture_1.get(4)))

# Defining video encoding (XVID)
fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')

i = 142

print('Recording starting.')
# Initializing the VideoWriter class - Try 15 fps
out_0 = cv2.VideoWriter(f'Dataset2/train/{i}.avi', fourcc, 10.0, cap_size0)
out_1 = cv2.VideoWriter(f'Dataset2/train/{i+1}.avi', fourcc, 10.0, cap_size1)

frameCount = 0

while True:
    # Reading frame-by-frame
    ret0, frame0 = video_capture_0.read()
    ret1, frame1 = video_capture_1.read()

    if (ret0 and ret1):
        # Displaying image
        cv2.imshow('Cam 0', frame0)
        cv2.imshow('Cam 1', frame1)
        # Writing image to file
        out_0.write(frame0)
        out_1.write(frame1)

        frameCount += 1
        print(frameCount)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        out_0.release()
        out_1.release()
        break

    if frameCount >= 80:
        break


# Release the capture & videowriter
video_capture_0.release()
video_capture_1.release()
cv2.destroyAllWindows()