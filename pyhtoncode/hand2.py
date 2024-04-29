import serial
import cv2
import mediapipe as mp

# Setting up the serial connection (Change the port name based on your configuration)
arduinoData = serial.Serial('/dev/tty.Bluetooth-Incoming-Port', 9600, timeout=1)  # Change port name accordingly

# MediaPipe hands setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

mp_drawing = mp.solutions.drawing_utils

# Function to send coordinates to Arduino
def send_coordinates_to_arduino(x, y, z):
    # Convert the coordinates to a string and send it to Arduino
    coordinates = f"{x},{y},{z}\r"
    arduinoData.write(coordinates.encode())
    print(f"Sent coordinates: X{x} Y{y} Z{z}\n")

# Video capture setup
capture = cv2.VideoCapture(0)

# Calibrate the Z-axis scaling (you need to adjust these based on your own testing and requirements)
z_reference = 200  # This might be a typical Z value when your hand is about 50 cm from the camera
z_scale = 500     # This is an arbitrary scaling factor to make Z roughly match the scale of X and Y

while True:
    success, frame = capture.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Flip the frame horizontally for a later selfie-view display, and convert the BGR image to RGB.
    frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
    results = hands.process(frame)

    # Draw the hand annotations on the frame.
    frame.flags.writeable = True
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
            h, w, _ = frame.shape
            x, y = int(wrist.x * w), int(wrist.y * h)
            z = wrist.z * z_scale + z_reference  # Apply scaling and offset to Z

            send_coordinates_to_arduino(x, y, int(z))
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Display the resulting frame
    cv2.imshow('MediaPipe Hands', frame)
    if cv2.waitKey(5) & 0xFF == 27:  # Press 'ESC' to exit
        break

# Clean up
hands.close()
capture.release()
cv2.destroyAllWindows()
arduinoData.close()