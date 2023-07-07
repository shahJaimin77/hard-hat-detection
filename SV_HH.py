import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2

model = torch.hub.load('ultralytics/yolov5', 'custom',
                       path='best_SH.pt', force_reload=True)

# Open the video file
cap = cv2.VideoCapture('D:/Sneha/Intership of Arunoday Tech/Safety Vest Detection/video3.mp4')

# Check if the file was opened successfully
if not cap.isOpened():
    print("Error opening video file")

# Loop over the frames in the video file
while cap.isOpened():
    # Read a frame from the video file
    ret, frame = cap.read()

    # Check if the frame was read successfully
    if not ret:
        break

    # Show the frame in a window

    results = model(frame)

    cv2.imshow('Safety Vest and HardHat Detection', np.squeeze(results.render()))

    # Wait for 25 milliseconds or until the user presses a key
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Release the video file and close all windows
cap.release()
cv2.destroyAllWindows()