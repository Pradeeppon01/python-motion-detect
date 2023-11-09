import cv2
import time


# Load the video file
video = cv2.VideoCapture('videoplayback123.mp4')
# Initialize variables
frame_count = 0
motion_detected = False

while True:
    # Read the next frame
    ret, frame = video.read()

    # Check if the frame was successfully read
    if not ret:
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (21, 21), 0)

    # If this is the first frame, initialize the reference frame
    if frame_count == 0:
        reference_frame = blurred

    # Calculate the absolute difference between the current frame and the reference frame
    frame_delta = cv2.absdiff(reference_frame, blurred)

    # Apply a threshold to the frame delta
    thresh = cv2.threshold(frame_delta, 30, 255, cv2.THRESH_BINARY)[1]

    # Dilate the thresholded image to fill in holes
    dilated = cv2.dilate(thresh, None, iterations=2)

    # Find contours of the dilated image
    contours, _ = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Check if any contours were found
    if len(contours) > 0:
        motion_detected = True
    else:
        motion_detected = False

    # Print the result every second
    if frame_count % video.get(cv2.CAP_PROP_FPS) == 0:
        if motion_detected:
            print(f"Motion detected at {frame_count // video.get(cv2.CAP_PROP_FPS)} seconds")
        else:
            print(f"Motion not detected at {frame_count // video.get(cv2.CAP_PROP_FPS)} seconds")

    # Increment the frame count
    frame_count += 1

# Release the video file and close all windows
video.release()
cv2.destroyAllWindows()

