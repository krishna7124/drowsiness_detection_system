from scipy.spatial import distance
from imutils import face_utils
import imutils
import dlib
import cv2
import time
import pygame


def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear


# Initialize pygame mixer
pygame.mixer.init()

# Load the sound files
alert_sound = pygame.mixer.Sound(
    "sound/alert.wav")  # Alert sound for eyes closed
# Sound to remind driver to take a rest
rest_sound = pygame.mixer.Sound("sound/rest.wav")

# Rest reminder interval in hours
rest_interval_hours = 0.01

# Total journey distance in kilometers
total_journey_distance_km = 1000

# Average speed assumption in km/h
average_speed_km_per_hour = 70

# Calculate total journey time in hours
total_journey_time_hours = total_journey_distance_km / average_speed_km_per_hour

alert_triggered = False
start_time = time.time()

thresh = 0.25
frame_check = 20
detect = dlib.get_frontal_face_detector()
# Dat file is the crux of the code
predict = dlib.shape_predictor("models/shape_predictor_68_face_landmarks.dat")

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]
cap = cv2.VideoCapture(0)
flag = 0
while True:
    ret, frame = cap.read()
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    subjects = detect(gray, 0)
    for subject in subjects:
        shape = predict(gray, subject)
        shape = face_utils.shape_to_np(shape)  # converting to NumPy Array
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
        if ear < thresh:
            flag += 1
            print(flag)
            if flag >= frame_check:
                cv2.putText(frame, "****************ALERT!****************", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.putText(frame, "****************ALERT!****************", (10, 325),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                print("Alert: Eyes closed for too long")
                if flag >= 4:
                    alert_triggered = True
                    # Play the alert sound
                    alert_sound.play()

        else:
            flag = 0
            if alert_triggered:
                alert_triggered = False
                # Stop the alert sound if it's playing
                pygame.mixer.stop()

        elapsed_time = time.time() - start_time
        # Check if rest reminder needs to be triggered
        if elapsed_time >= rest_interval_hours * 3600:
            if not alert_triggered:
                print("Alert: Please take a rest")
                rest_sound.play()
                # Reset start time for rest interval
                start_time = time.time()

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cv2.destroyAllWindows()
cap.release()
