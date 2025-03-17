import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import time
import logging
import threading
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import register_keras_serializable
import speech_recognition as sr  # For speech-to-text functionality
import face_recognition         # For face detection and recognition
import ctypes                   # For locking the workstation on Windows
import screen_brightness_control as sbc
import webbrowser  # Needed to open the browser


google_search_stt_active = False
google_search_query = ""
google_search_thread = None

# Suppress specific warnings and set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


@register_keras_serializable()
class BackgroundSubtractionLayer(tf.keras.layers.Layer):
    def __init__(self, background, **kwargs):
        super(BackgroundSubtractionLayer, self).__init__(**kwargs)
        self.background = tf.constant(background, dtype=tf.float32)

    def call(self, inputs):
        return inputs - self.background

    def get_config(self):
        config = super().get_config().copy()
        config.update({'background': self.background.numpy().tolist()})
        return config

    @classmethod
    def from_config(cls, config):
        background = np.array(config.pop('background'))
        return cls(background=background, **config)


logging.info("Loading hand gesture model...")
model = load_model('my_model.h5', custom_objects={'BackgroundSubtractionLayer': BackgroundSubtractionLayer})


logging.info("Initializing webcam...")
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

logging.info("Initializing MediaPipe...")
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.3, max_num_hands=2)


labels_dict = {
    0: 'STT',  
    31: 'SU',
    30: 'SD',
    27: 'VU',
    28: 'VD',
    26: 'Left',
    1: 'Right',
    6: 'Close',
    7: 'Minimize',
    5: 'Next',
    22: 'LH',
    10: 'DLC',
    8: 'FaceAuth',
    29: 'Enter',
    23: 'BU',
    24: 'BD',
    17: 'GoogleSearch'

}

# -------------------------------------------------
# Define Gesture Actions
# -------------------------------------------------
def left_click():
    pyautogui.click()

def right_click():
    pyautogui.click(button='right')

def volume_up():
    pyautogui.press('volumeup')

def volume_down():
    pyautogui.press('volumedown')

def scroll_up():
    # Total scroll amount (adjust as needed)
    total_scroll = 100
    # Number of increments for smoother effect
    increments = 10
    for _ in range(increments):
        pyautogui.scroll(total_scroll // increments)
        time.sleep(0.01)  # Adjust the pause duration for smoothness

def scroll_down():
    total_scroll = 100
    increments = 10
    for _ in range(increments):
        pyautogui.scroll(- (total_scroll // increments))
        time.sleep(0.01)

def close_window():
    pyautogui.hotkey('alt', 'f4')

def minimize_window():
    pyautogui.hotkey('win', 'down')

def next_window():
    pyautogui.hotkey('alt', 'shift', 'tab')

def left_click_hold():
    pyautogui.mouseDown()

def left_click_release():
    pyautogui.mouseUp()

def double_left_click():
    pyautogui.doubleClick()

def copy():
    pyautogui.hotkey('ctrl', 'c')

def paste():
    pyautogui.hotkey('ctrl', 'v')

def brightness_up():
    try:
        # Get the current brightness; assuming the primary display (index 0)
        current_brightness = sbc.get_brightness()[0]
        new_brightness = min(current_brightness + 10, 100)
        sbc.set_brightness(new_brightness)
        logging.info(f"Brightness increased to {new_brightness}%")
    except Exception as e:
        logging.error(f"Error increasing brightness: {e}")

def brightness_down():
    try:
        # Get the current brightness; assuming the primary display (index 0)
        current_brightness = sbc.get_brightness()[0]
        new_brightness = max(current_brightness - 10, 0)
        sbc.set_brightness(new_brightness)
        logging.info(f"Brightness decreased to {new_brightness}%")
    except Exception as e:
        logging.error(f"Error decreasing brightness: {e}")

def enter():
    pyautogui.press('enter')

def google_search_listener():
    """
    A dedicated listener for Google search that continuously appends
    recognized speech to the global query string until deactivated.
    """
    global google_search_stt_active, google_search_query
    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    
    # Calibrate microphone to ambient noise
    with mic as source:
        recognizer.adjust_for_ambient_noise(source)
    
    logging.info("Google Search STT listener active. Please speak your search query.")
    while google_search_stt_active:
        with mic as source:
            try:
                # Listen for a short phrase; adjust timeout/phrase_time_limit as needed.
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=5)
            except sr.WaitTimeoutError:
                continue  # No speech detected within timeout period
        
        try:
            text = recognizer.recognize_google(audio)
            logging.info(f"Google STT recognized: {text}")
            # Append recognized text (with a space) to the query
            google_search_query += " " + text
        except sr.UnknownValueError:
            logging.info("Google STT could not understand audio.")
        except sr.RequestError as e:
            logging.error(f"Google STT error: {e}")

def google_search_handler():
    """
    This function toggles the Google search speech-to-text mode.
    
    - If the listener is not active, it starts listening.
    - If the listener is already active, it stops listening,
      uses the accumulated query to perform a Google search,
      and then resets the query.
    """
    global google_search_stt_active, google_search_query, google_search_thread
    if not google_search_stt_active:
        # Start STT for Google search
        google_search_query = ""  # Reset the query
        google_search_stt_active = True
        google_search_thread = threading.Thread(target=google_search_listener, daemon=True)
        google_search_thread.start()
        logging.info("Google search STT started. Speak your query.")
    else:
        # Stop STT and perform Google search
        google_search_stt_active = False
        if google_search_thread is not None:
            google_search_thread.join(timeout=1)
        
        # Prepare and perform the search if any query was captured
        query = google_search_query.strip()
        if query:
            search_url = "https://www.google.com/search?q=" + query.replace(" ", "+")
            webbrowser.open_new_tab(search_url)
            logging.info(f"Performing Google search for: {query}")
        else:
            logging.info("No query recognized for Google search.")
        
        # Reset the query for the next round
        google_search_query = ""

speech_to_text_active = False  # Global flag for speech-to-text mode
speech_thread = None           # Thread for listening in the background

def speech_to_text_listener():
    """Continuously listen to the microphone and type recognized speech."""
    global speech_to_text_active
    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    
    # Calibrate the microphone to the ambient noise level
    with mic as source:
        recognizer.adjust_for_ambient_noise(source)
    
    logging.info("Speech-to-text listener active.")
    while speech_to_text_active:
        with mic as source:
            try:
                # Listen for up to 5 seconds; adjust phrase_time_limit as needed.
                audio = recognizer.listen(source, timeout=5, phrase_time_limit=5)
            except sr.WaitTimeoutError:
                continue  # No speech detected within the timeout period
        try:
            text = recognizer.recognize_google(audio)
            logging.info(f"Recognized speech: {text}")
            # Type the recognized text with a trailing space
            pyautogui.write(text + " ")
        except sr.UnknownValueError:
            logging.info("Speech recognition could not understand audio.")
        except sr.RequestError as e:
            logging.error(f"Could not request results from the speech recognition service; {e}")

def toggle_speech_to_text():
    """Toggle the speech-to-text mode on and off."""
    global speech_to_text_active, speech_thread
    if not speech_to_text_active:
        logging.info("Starting speech-to-text mode...")
        speech_to_text_active = True
        speech_thread = threading.Thread(target=speech_to_text_listener, daemon=True)
        speech_thread.start()
    else:
        logging.info("Stopping speech-to-text mode...")
        speech_to_text_active = False
        if speech_thread is not None:
            speech_thread.join(timeout=1)


def face_recognition_module():

    logging.info("Starting face recognition module...")
    authorized = False

    # Load the authorized face encoding
    try:
        authorized_image = face_recognition.load_image_file("authorized_faces/authorized_face.jpg")
        authorized_encoding = face_recognition.face_encodings(authorized_image)[0]
    except Exception as e:
        logging.error(f"Error loading authorized face: {e}")
        return

    cap_face = cv2.VideoCapture(0)
    start_time = time.time()

    while True:
        ret, frame = cap_face.read()
        if not ret:
            logging.error("Failed to read from webcam for face recognition.")
            break

        # Convert frame to RGB for face_recognition processing
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces([authorized_encoding], face_encoding)
            if any(matches):
                authorized = True
                break

        # Display the face recognition window
        cv2.imshow("Face Recognition", frame)

        # Exit conditions: authorized face found or timeout after 5 seconds
        if authorized or (time.time() - start_time > 5):
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap_face.release()
    cv2.destroyWindow("Face Recognition")

    if authorized:
        logging.info("Authorized face recognized. Locking laptop...")
        # Lock the workstation (Windows-specific)
        ctypes.windll.user32.LockWorkStation()
    else:
        logging.info("Authorized face not recognized within 5 seconds. Resuming gesture control.")

def trigger_face_auth():
    """Helper function to trigger face authentication by temporarily stopping hand gesture capture."""
    global cap
    logging.info("Triggering face authentication...")
    cap.release()  # Release the hand gesture webcam capture
    face_recognition_module()
    # Reinitialize webcam for hand gesture recognition
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)


gesture_to_action = {
    'STT': toggle_speech_to_text,  # Toggle speech recognition for the STT gesture
    'SU': scroll_up,
    'SD': scroll_down,
    'VU': volume_up,
    'Left': left_click,
    'VD': volume_down,
    'Right': right_click,
    'Close': close_window,
    'Minimize': minimize_window,
    'Next': next_window,
    'LH': left_click_hold,
    'DLC': double_left_click,
    'Copy': copy,
    'Paste': paste,
    'BU': brightness_up,
    'BD': brightness_down,
    'FaceAuth': trigger_face_auth,
    'Enter': enter,
    'GoogleSearch': google_search_handler
}


gesture_last_time = {}  # To enforce a delay between repeated actions
gesture_delay = 1  # seconds

# For mouse movement smoothing
screen_width, screen_height = pyautogui.size()
cursor_x_buffer = []
cursor_y_buffer = []
buffer_size = 5

def is_index_finger_extended(hand_landmarks):
    extended = [False] * 5
    for i, tip in enumerate([8, 12, 16, 20]):
        if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:
            extended[i] = True
    return extended[0] and not any(extended[1:])

# -------------------------------------------------
# Main Loop: Capture Frames, Process Gestures, and Trigger Actions
# -------------------------------------------------
try:
    logging.info("Starting main loop...")
    while True:
        ret, frame = cap.read()
        if not ret:
            logging.error("Webcam disconnected. Reconnecting...")
            cap.release()
            time.sleep(2)
            cap = cv2.VideoCapture(0)
            continue

        H, W, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        gestures_in_frame = set()  # Keep track of unique gestures in the current frame

        if results.multi_hand_landmarks:
            for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

                # Prepare input for the CNN model
                x_coords = [lm.x for lm in hand_landmarks.landmark]
                y_coords = [lm.y for lm in hand_landmarks.landmark]
                data_aux = []
                for lm in hand_landmarks.landmark:
                    data_aux.extend([lm.x - min(x_coords), lm.y - min(y_coords)])
                try:
                    input_data = np.array(data_aux[:42]).reshape(1, 7, 6, 1)
                    prediction = model.predict(input_data)
                    predicted_index = np.argmax(prediction)
                    predicted_character = labels_dict.get(predicted_index, 'Unknown')
                    gestures_in_frame.add(predicted_character)
                    logging.info(f"Detected gesture: {predicted_character}")
                except Exception as e:
                    logging.error(f"Error in prediction for hand {hand_idx}: {e}")
                    continue

                # If only the index finger is extended, update cursor position
                if is_index_finger_extended(hand_landmarks):
                    index_finger_tip = hand_landmarks.landmark[8]
                    cursor_x = int(index_finger_tip.x * screen_width)
                    cursor_y = int(index_finger_tip.y * screen_height)
                    cursor_x_buffer.append(cursor_x)
                    cursor_y_buffer.append(cursor_y)
                    if len(cursor_x_buffer) > buffer_size:
                        cursor_x_buffer.pop(0)
                        cursor_y_buffer.pop(0)
                    smoothed_x = int(np.mean(cursor_x_buffer))
                    smoothed_y = int(np.mean(cursor_y_buffer))
                    # Invert the coordinates relative to the screen dimensions:
                    inverted_x = screen_width - smoothed_x
                    inverted_y = screen_height - smoothed_y  # For vertical inversion; remove if not desired
                    # If you only want horizontal inversion, you can keep smoothed_y:
                    # pyautogui.moveTo(inverted_x, smoothed_y, duration=0.01)
                    pyautogui.moveTo(inverted_x, inverted_y, duration=0.01)

        # Trigger actions for detected gestures (with a delay to prevent rapid repeats)
        current_time = time.time()
        for gesture in gestures_in_frame:
            if gesture not in gesture_last_time or current_time - gesture_last_time[gesture] > gesture_delay:
                if gesture in gesture_to_action:
                    gesture_to_action[gesture]()
                    logging.info(f"Action performed for gesture: {gesture}")
                gesture_last_time[gesture] = current_time

        # Display the control frame
        cv2.imshow('Hand Gesture Control', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            logging.info("Exiting program.")
            break
finally:
    logging.info("Releasing resources...")
    cap.release()
    cv2.destroyAllWindows()