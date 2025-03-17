import numpy as np
import cv2
import time

# -----------------------------
# Open Webcam
# -----------------------------
video = cv2.VideoCapture(0)  # 0 is the default webcam

if not video.isOpened():
    print("Error: Could not open webcam")
    exit()

# Get frame dimensions
width  = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
print("Frame dimensions: {}x{}".format(width, height))

# -----------------------------
# ViBe Parameters
# -----------------------------
N = 20                  # Number of samples per pixel in the background model
R_val = 20              # Matching threshold (radius)
Number_min = 2          # Minimum number of matches needed to consider a pixel background
Rand_Samples = 20       # Background model update probability: 1 in Rand_Samples chance
Pad = 10                # Padding for drawing bounding boxes

# -----------------------------
# Initialize the Background Model
# -----------------------------
# Capture the first frame (make sure your hand is NOT in the frame at initialization)
ret, frame_col = video.read()
if not ret:
    print("Error: Could not read frame from webcam.")
    exit()

frame_gray = cv2.cvtColor(frame_col, cv2.COLOR_BGR2GRAY)

# Create a background model with N noisy versions of the first frame.
Background_Model = np.empty((height, width, N), dtype=np.uint8)
for n in range(N):
    noise = np.random.randint(-10, 11, size=(height, width), dtype=np.int16)
    model_frame = frame_gray.astype(np.int16) + noise
    Background_Model[:, :, n] = np.clip(model_frame, 0, 255).astype(np.uint8)

# -----------------------------
# Main Loop: Process Each Frame
# -----------------------------
while True:
    start_time = time.time()

    ret, frame_col = video.read()
    if not ret:
        break

    # Convert the frame to grayscale for background subtraction.
    frame_gray = cv2.cvtColor(frame_col, cv2.COLOR_BGR2GRAY)
    # Replicate the current frame along a new third axis so it can be compared with the background model.
    frame_3D = np.repeat(frame_gray[:, :, np.newaxis], N, axis=2)

    # -----------------------------
    # Background Subtraction (ViBe)
    # -----------------------------
    # Compute the absolute difference between each background sample and the current frame.
    diff = np.abs(Background_Model.astype(np.int16) - frame_3D.astype(np.int16))
    # For each pixel in each channel, check if the difference is less than the threshold.
    matches = diff < R_val  
    # Count the number of matching samples at each pixel.
    match_count = np.sum(matches, axis=2)
    # If enough samples match, mark the pixel as background (0); otherwise, foreground (255).
    Segmentation = np.where(match_count >= Number_min, 0, 255).astype(np.uint8)

    # -----------------------------
    # Update the Background Model (only for pixels marked as background)
    # -----------------------------
    bg_mask = (Segmentation == 0)

    # Self-update: with a probability of 1/Rand_Samples, update the pixel's own background model.
    update_self = (np.random.randint(0, Rand_Samples, size=(height, width)) == 0) & bg_mask
    rand_channel = np.random.randint(0, N, size=(height, width))
    ys, xs = np.where(update_self)
    Background_Model[ys, xs, rand_channel[ys, xs]] = frame_gray[ys, xs]

    # Neighbor-update: with the same probability, update a random neighbor's model.
    update_neighbor = (np.random.randint(0, Rand_Samples, size=(height, width)) == 0) & bg_mask
    offset_y = np.random.randint(-1, 2, size=(height, width))
    offset_x = np.random.randint(-1, 2, size=(height, width))
    Y, X = np.indices((height, width))
    neighbor_Y = np.clip(Y + offset_y, 0, height - 1)
    neighbor_X = np.clip(X + offset_x, 0, width - 1)
    rand_channel2 = np.random.randint(0, N, size=(height, width))
    ys2, xs2 = np.where(update_neighbor)
    Background_Model[neighbor_Y[ys2, xs2], neighbor_X[ys2, xs2], rand_channel2[ys2, xs2]] = frame_gray[ys2, xs2]

    # Apply a median filter to reduce noise in the segmentation mask.
    Segmentation = cv2.medianBlur(Segmentation, 7)

    # -----------------------------
    # Skin Color Detection (to help identify the hand)
    # -----------------------------
    # Convert the current frame to YCrCb color space.
    frame_YCrCb = cv2.cvtColor(frame_col, cv2.COLOR_BGR2YCrCb)
    # Define thresholds for skin color (these values can be adjusted).
    lower_skin = np.array([0, 133, 77], dtype=np.uint8)
    upper_skin = np.array([255, 173, 127], dtype=np.uint8)
    skin_mask = cv2.inRange(frame_YCrCb, lower_skin, upper_skin)
    
    # -----------------------------
    # Combine Background Subtraction and Skin Detection
    # -----------------------------
    # Only consider regions that are both foreground (from ViBe) and within the skin color range.
    hand_mask = cv2.bitwise_and(Segmentation, skin_mask)
    # Optional: apply median filtering to smooth the hand mask.
    hand_mask = cv2.medianBlur(hand_mask, 7)

    # -----------------------------
    # Find and Draw Contours Around the Hand
    # -----------------------------
    contours, _ = cv2.findContours(hand_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w > 10 and h > 10:
            cv2.rectangle(frame_col, (x - Pad, y - Pad), (x + w + Pad, y + h + Pad), (0, 255, 0), 2)

    # -----------------------------
    # Display the Results
    # -----------------------------
    cv2.imshow("Webcam Feed (with Hand Detection)", frame_col)
    cv2.imshow("Hand Mask (Foreground & Skin Combined)", hand_mask)
    # (Optional) Show the individual masks for debugging:
    cv2.imshow("Foreground Segmentation", Segmentation)
    cv2.imshow("Skin Mask", skin_mask)
    
    elapsed = time.time() - start_time
    print("Frame processing time: {:.3f} sec".format(elapsed))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
video.release()
cv2.destroyAllWindows()
