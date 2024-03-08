import cv2
import numpy as np

# Define the preprocessing function
def preprocess_frame(frame, target_size=(224, 224)):
    frame_resized = cv2.resize(frame, target_size)
    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    frame_normalized = frame_rgb / 255.0
    return np.expand_dims(frame_normalized, axis=0)
def calculate_dominant_color(frame):
    # Convert the frame to HSV color space
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Calculate the histogram of the hue channel
    hist_hue = cv2.calcHist([hsv_frame], [0], None, [180], [0, 180])
    
    # Find the index of the most frequent hue value
    dominant_hue_index = np.argmax(hist_hue)
    
    # Convert the dominant hue index to its corresponding color
    dominant_hue_color = np.uint8([[[dominant_hue_index, 255, 255]]])
    dominant_color_rgb = cv2.cvtColor(dominant_hue_color, cv2.COLOR_HSV2BGR)[0][0]
    
    return dominant_color_rgb

# Load the video using OpenCV
video_path = 'C:/Users/money/Downloads/BOTB/Download-3.mp4'
cap = cv2.VideoCapture(video_path)
print(f"Video Capture Object Opened: {cap.isOpened()}")

# Initialize a list to store preprocessed frames
preprocessed_frames = []

# Define the color of the endzone
endzone_color_range = np.array([[120, 115, 63], [128, 125, 77]])  # RGB color of the endzone

# Iterate through the video frames and preprocess each frame
frame_count = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    preprocessed_frame = preprocess_frame(frame)
    preprocessed_frames.append(preprocessed_frame)
    cv2.imshow('Frame', frame)
    # Apply any additional processing here
    h, w, _ = frame.shape
    print("Processing frame...")
    for x in range(h):
        for y in range(w):
            pixel_color = tuple(frame[x, y])
            
            if np.all(np.logical_and(pixel_color >= endzone_color_range[0], pixel_color <= endzone_color_range[1])):
                print(f'Endzone color detected at Coordinates: ({x}, {y}), Color: {pixel_color}')
                break
    dominant_color = calculate_dominant_color(frame)
    
    # Print the frame count and dominant color
    print(f'Frame {frame_count}: Dominant Color (BGR): {dominant_color}')
    frame_count +=1
# Convert the list of preprocessed frames to a NumPy array
preprocessed_frames = np.array(preprocessed_frames)
print('Shape of preprocessed frames:', preprocessed_frames.shape)

# Release the video capture object
cap.release()



    


