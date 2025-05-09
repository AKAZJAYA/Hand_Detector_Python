import os
import cv2
import re
import numpy as np
import mediapipe as mp

# Variables
width, height = 1280, 720
folderPath = "Presentation"

# Camera setup
cap = cv2.VideoCapture(0)
cap.set(3, width)
cap.set(4, height)

# Get the List of Presentation Images
def get_image_number(filename):
    match = re.search(r'(\d+)', filename)
    return int(match.group(1)) if match else 0

# Check if folder exists and contains files
if not os.path.exists(folderPath):
    os.makedirs(folderPath)
    print(f"Created empty folder: {folderPath}")
    print("Please add presentation images to this folder and restart the application")
    exit()

pathImages = [f for f in sorted(os.listdir(folderPath), key=get_image_number)
              if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

if not pathImages:
    print(f"No images found in {folderPath}. Please add images and restart.")
    exit()

print(f"Found images: {pathImages}")

# Variables
imgNumber = 0
hs, ws = int(120 * 3), int(213 * 3)

# Set up MediaPipe directly - avoiding cvzone's wrapper to have more control
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands_detector = mp_hands.Hands(static_image_mode=False,
                                max_num_hands=1,
                                min_detection_confidence=0.7,
                                min_tracking_confidence=0.7)

class HandDetector:
    def __init__(self, hands_detector):
        self.hands_detector = hands_detector
        
    def fingersUp(self, hand_info):
        """
        Returns a list of which fingers are up
        [thumb, index, middle, ring, pinky]
        1 means finger is up, 0 means finger is down
        """
        fingers = []
        # Thumb (based on horizontal position)
        # If thumb tip x position is to the right of thumb MCP for right hand
        lmList = hand_info["lmList"]
        
        # Check thumb by comparing x-coordinate (horizontal)
        if lmList[4][0] > lmList[3][0]:
            fingers.append(1)
        else:
            fingers.append(0)
            
        # For other 4 fingers, check if fingertip y is above finger pip
        for id in range(1, 5):
            # If fingertip y position is above finger pip y position (lower y value means higher up in image)
            if lmList[8+4*(id-1)][1] < lmList[6+4*(id-1)][1]:
                fingers.append(1)
            else:
                fingers.append(0)
                
        return fingers

# Create detector instance
detector = HandDetector(hands_detector)

def process_hand_landmarks(image, hand_landmarks):
    """Process hand landmarks and return hand information"""
    h, w, c = image.shape
    hand_info = {}
    landmarks_list = []
    x_coords = []
    y_coords = []

    for id, lm in enumerate(hand_landmarks.landmark):
        px, py, pz = int(lm.x * w), int(lm.y * h), int(lm.z * w)
        landmarks_list.append([px, py, pz])
        x_coords.append(px)
        y_coords.append(py)

    # Calculate bounding box
    x_min, x_max = min(x_coords), max(x_coords)
    y_min, y_max = min(y_coords), max(y_coords)
    box_w, box_h = x_max - x_min, y_max - y_min
    bbox = (x_min, y_min, box_w, box_h)
    center = (x_min + (box_w // 2), y_min + (box_h // 2))

    hand_info["lmList"] = landmarks_list
    hand_info["bbox"] = bbox
    hand_info["center"] = center

    return hand_info

while True:
    # Check camera
    if not cap.isOpened():
        print("Error: Could not open camera.")
        break

    # Get frame
    success, img = cap.read()
    img = cv2.flip(img, 1)
    if not success or img is None:
        print("Failed to grab frame from camera")
        continue

    try:
        # Make sure image is properly formatted for MediaPipe
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # MediaPipe requires contiguous arrays for processing
        img_rgb = np.ascontiguousarray(img_rgb)

        # Ensure img_rgb is a valid numpy array of uint8 type
        if not isinstance(img_rgb, np.ndarray) or img_rgb.dtype != np.uint8:
            print(f"Invalid image format: {type(img_rgb)}, dtype: {img_rgb.dtype}")
            continue

        # Process image with MediaPipe
        results = hands_detector.process(img_rgb)

        # Process results
        all_hands = []

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Process hand landmarks
                hand_info = process_hand_landmarks(img, hand_landmarks)
                all_hands.append(hand_info)

                # Draw landmarks
                mp_drawing.draw_landmarks(
                    img,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

                # Draw bounding box
                bbox = hand_info["bbox"]
                cv2.rectangle(img,
                              (bbox[0] - 20, bbox[1] - 20),
                              (bbox[0] + bbox[2] + 20, bbox[1] + bbox[3] + 20),
                              (255, 0, 255), 2)
            
            # If hands are detected, check fingers up status
            if all_hands:
                hand = all_hands[0]
                fingers = detector.fingersUp(hand)
                print(fingers)
                
                # Draw finger status on image
                finger_names = ["Thumb", "Index", "Middle", "Ring", "Pinky"]
                finger_text = ""
                for i, finger in enumerate(fingers):
                    if finger == 1:
                        finger_text += f"{finger_names[i]}: UP, "
                    else:
                        finger_text += f"{finger_names[i]}: DOWN, "
                
                cv2.putText(img, finger_text[:-2], (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Handle presentation slides
        if 0 <= imgNumber < len(pathImages):
            pathFullImage = os.path.join(folderPath, pathImages[imgNumber])
            imgCurrent = cv2.imread(pathFullImage)

            if imgCurrent is not None:
                # Adding webcam image on the slides
                imgSmall = cv2.resize(img, (ws, hs))
                h, w, _ = imgCurrent.shape

                # Make sure dimensions are valid
                if h >= hs and w >= ws:
                    imgCurrent[0:hs, w - ws:w] = imgSmall
                    cv2.imshow("Slides", imgCurrent)
                else:
                    print(f"Slide image too small: {imgCurrent.shape}")
        else:
            print(f"Invalid image index: {imgNumber}")

        # Check for gesture to change slides (can be implemented based on hand position)

        # Display the webcam
        cv2.imshow("Image", img)

    except Exception as e:
        print(f"Error processing frame: {e}")

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

# Clean up
hands_detector.close()
cap.release()
cv2.destroyAllWindows()