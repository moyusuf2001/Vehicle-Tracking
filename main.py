# Import necessary libraries
import cv2
import numpy as np
from object_detection import ObjectDetection
import math
import os

# Initialize Object Detection
od = ObjectDetection()

# Open a video file for capturing frames
cap = cv2.VideoCapture("1.mp4")

# Get the width, height, and frames per second (FPS) of the input video
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Create a VideoWriter object to save the output video
output_path = os.path.join(os.path.dirname("1.mp4"), 'output_video.mp4')
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4 format
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# Initialize count to keep track of the frame number
count = 0

# List to store the center points of objects in the previous frame
center_points_prev_frame = []

# Dictionary to track objects using unique IDs
tracking_objects = {}

# Variable to keep track of the unique ID assigned to each object
track_id = 0

# Start an infinite loop for processing each frame
while True:
    # Read a frame from the video capture
    ret, frame = cap.read()  # ret is a boolean indicator

    # Increment the frame count
    count += 1

    # Break the loop if the frame was not successfully read (end of video)
    if not ret:
        break

    # List to store the center points of objects in the current frame
    center_points_cur_frame = []

    # Detect objects in the current frame using the ObjectDetection class
    (class_ids, scores, boxes) = od.detect(frame)

    # Loop over the detected boxes
    for box in boxes:
        # Extract coordinates of the bounding box
        (x, y, w, h) = box

        # Calculate the center point of the bounding box
        cx = int((x + x + w) / 2)
        cy = int((y + y + h) / 2)

        # Append the center point to the list
        center_points_cur_frame.append((cx, cy))

        # Draw a rectangle around the detected object on the frame
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Check if it's the first or second frame
    if count <= 2:
        # Associate object IDs based on proximity in the first two frames
        for pt in center_points_cur_frame:
            for pt2 in center_points_prev_frame:
                distance = math.hypot(pt2[0] - pt[0], pt2[1] - pt[1])  # Euclidean distance

                if distance < 20:
                    tracking_objects[track_id] = pt
                    track_id += 1
    else:
        # Update object IDs based on their proximity in the current and previous frames
        tracking_objects_copy = tracking_objects.copy()
        center_points_cur_frame_copy = center_points_cur_frame.copy()

        for object_id, pt2 in tracking_objects_copy.items():
            object_exists = False
            for pt in center_points_cur_frame_copy:
                distance = math.hypot(pt2[0] - pt[0], pt2[1] - pt[1])

                if distance < 20:
                    tracking_objects[object_id] = pt
                    object_exists = True
                    if pt in center_points_cur_frame:
                        center_points_cur_frame.remove(pt)
                    continue

            # Remove IDs of objects that are lost
            if not object_exists:
                tracking_objects.pop(object_id)

        # Add new IDs for newly found objects
        for pt in center_points_cur_frame:
            tracking_objects[track_id] = pt
            track_id += 1

    # Draw circles and labels for each tracked object on the frame
    for object_id, pt in tracking_objects.items():
        cv2.circle(frame, pt, 5, (0, 0, 255), -1)
        cv2.putText(frame, str(object_id), (pt[0], pt[1] - 7), 0, 1, (0, 0, 255), 2)

    # Print the currently tracked objects and their positions
    print("Tracking objects")
    print(tracking_objects)

    # Print the center points of objects in the current frame
    print("CUR FRAME LEFT PTS")
    print(center_points_cur_frame)

    # Write the processed frame to the output video
    out.write(frame)

    # Display the frame with tracking information
    cv2.imshow("Frame", frame)

    # Make a copy of the center points for the next iteration
    center_points_prev_frame = center_points_cur_frame.copy()

    # Wait for a key event, and break the loop if the 'Esc' key is pressed
    key = cv2.waitKey(1)
    if key == 27:
        break

# Release the video capture and writer objects
cap.release()
out.release()

# Close all OpenCV windows
cv2.destroyAllWindows()
