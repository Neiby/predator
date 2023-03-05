import cv2
import sys

infile = sys.argv[1]

# Open the video file
video = cv2.VideoCapture(infile)

# Initialize the first frame
_, frame1 = video.read()
gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
gray1 = cv2.GaussianBlur(gray1, (5, 5), 0)

# Set the threshold for motion detection
threshold = 3

# Initialize variables for controlling the video playback
skip_frames = 10  # Number of frames to skip forward or backward
paused = False

# Initialize variables for keeping track of motion detection boxes
motion_boxes = []
motion_box_duration = 12  # Number of frames to keep the boxes on the screen
min_box_width = 20  # Minimum width of motion detection boxes
min_box_height = 20  # Minimum height of motion detection boxes


while True:
    # Read the current frame
    ret, frame2 = video.read()
    if not ret:
        break

    # Convert the frame to grayscale and blur it
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.GaussianBlur(gray2, (5, 5), 0)

    # Calculate the absolute difference between the current frame and the first frame
    frame_diff = cv2.absdiff(gray1, gray2)

    # Apply a threshold to the frame difference
    _, threshold_diff = cv2.threshold(frame_diff, threshold, 255, cv2.THRESH_BINARY)

    # Find the contours in the thresholded frame difference
    contours, _ = cv2.findContours(threshold_diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Add the motion detection boxes to the list of boxes to be drawn on the current frame
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if w >= min_box_width and h >= min_box_height:
            motion_boxes.append((x, y, w, h, motion_box_duration))

    # Draw the motion detection boxes on the current frame
    for box in motion_boxes:
        x, y, w, h, duration = box
        cv2.rectangle(frame2, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Decrement the duration of each motion detection box
    for i, box in enumerate(motion_boxes):
        x, y, w, h, duration = box
        duration -= 1
        if duration <= 0:
            motion_boxes.pop(i)
        else:
            motion_boxes[i] = (x, y, w, h, duration)

    # Show the frame with the motion detection boxes
    cv2.imshow('Motion Detection', frame2)

    # Check for keyboard input
    key = cv2.waitKey(25)

    # Pause or resume the video playback when the space bar is pressed
    if key == ord(' '):
        paused = not paused

    # Skip forward in the video when the right arrow key is pressed
    if key == ord('d'):
        for i in range(skip_frames):
            video.read()

    # Skip backward in the video when the left arrow key is pressed
    if key == ord('a'):
        # Go back to the beginning of the video if the skip amount is greater than the current frame number
        if video.get(cv2.CAP_PROP_POS_FRAMES) < skip_frames:
            video.set(cv2.CAP_PROP_POS_FRAMES, 0)
        else:
            video.set(cv2.CAP_PROP_POS_FRAMES, video.get(cv2.CAP_PROP_POS_FRAMES) - skip_frames)

    # Quit the program when the 'q' key is pressed
    if key == ord('q'):
        break

    # Update the first frame if the video is not paused
    if not paused:
        gray1 = gray2.copy()

# Release the video file and close the windows
video.release()
cv2.destroyAllWindows()
