import cv2

# Input video file
input_video_path = 'TM06202003230529110736780.mp4'

# Initialize video capture
video_capture = cv2.VideoCapture(input_video_path)

# Initialize variables for motion detection
previous_frame = None
current_time = 0

while True:
    ret, frame = video_capture.read()

    if not ret:
        break

    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if previous_frame is None:
        previous_frame = gray_frame
        continue

    # Calculate the absolute difference between the current and previous frame
    frame_diff = cv2.absdiff(previous_frame, gray_frame)

    # Set a threshold for detecting motion
    threshold = 30
    motion_detected = frame_diff.mean() > threshold

    # Check if one second has passed
    if int(video_capture.get(cv2.CAP_PROP_POS_MSEC)) // 1000 > current_time:
        if motion_detected:
            print(f'{current_time}.{int(video_capture.get(cv2.CAP_PROP_POS_MSEC)) // 100}: Motion detected')
        else:
            print(f'{current_time}.{int(video_capture.get(cv2.CAP_PROP_POS_MSEC)) // 100}: No motion')
        current_time += 1

    previous_frame = gray_frame

# Release the video capture
video_capture.release()

