import cv2
import mediapipe as mp
import serial
import time
from collections import deque

arduinoData = serial.Serial('/dev/cu.usbserial-0001', 9600) #Change this based on your serial port

# Define a deque with a maximum length of 10 for smoothing
smooth_y = deque(maxlen=10)

def map_value(value, leftMin, leftMax, rightMin, rightMax):
    # Maps a value from one range to another range
    leftSpan = leftMax - leftMin
    rightSpan = rightMax - rightMin
    valueScaled = float(value - leftMin) / float(leftSpan)
    return rightMin + (valueScaled * rightSpan)

def send_coordinates_to_arduino(x, y):
    # Convert the coordinates to a string and send it to Arduino
    inverted_y = capture.get(cv2.CAP_PROP_FRAME_HEIGHT) - y

    # Add the new y-coordinate to the deque
    smooth_y.append(inverted_y)

    # Calculate the average of the last 10 y-coordinates for smoothing
    smoothed_y = sum(smooth_y) / len(smooth_y)

    # Map the y-coordinate to the range expected by the servo motor
    mapped_y = map_value(smoothed_y, 0, capture.get(cv2.CAP_PROP_FRAME_HEIGHT), 0, 360)

    # Ensure the y-coordinate is always positive
    positive_y = abs(mapped_y)

    # Map the y-coordinate to the same range as the x-coordinate
    mapped_y = map_value(positive_y, 0, 360, 0, 2048)

    # Convert the coordinates to a string and send it to Arduino
    coordinates = f"{x},{mapped_y}\r"
    arduinoData.write(coordinates.encode())
    print(f"X{x}Y{mapped_y}\n")

capture = cv2.VideoCapture(0)

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

# Set the delay between each coordinate send
delay = 100  # milliseconds

with mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    last_send_time = time.time() * 1000  # Initialize to a time that allows the first send
    while capture.isOpened():
        isTrue, frame = capture.read()
        if not isTrue:
            print("Ignoring empty camera frame.")
            continue

        # Flip the image horizontally for a later selfie-view display, and convert the BGR image to RGB.
        frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
        # To improve performance, optionally mark the image as not writeable to pass by reference.
        frame.flags.writeable = False
        results = hands.process(frame)

        # Draw the hand annotations on the image.
        frame.flags.writeable = True
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                # Send the coordinates of the center of the hand (landmark 0) to the Arduino
                hand_x = int(hand_landmarks.landmark[0].x * frame.shape[1])
                hand_y = int(hand_landmarks.landmark[0].y * frame.shape[0])
                # Scale the coordinates to increase the range of movement
                hand_x = hand_x * 2
                hand_y = hand_y * 2
                current_time = time.time() * 1000
                if current_time - last_send_time >= delay:
                    send_coordinates_to_arduino(hand_x, hand_y)
                    last_send_time = current_time

        cv2.imshow('MediaPipe Hands', frame)
        if cv2.waitKey(10) & 0xFF == ord('d'):
            break

capture.release()
cv2.destroyAllWindows()