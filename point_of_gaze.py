# Import libraries
import cv2
import mediapipe as mp
from pupil_module import PupilTracker

# Instantiate mediapipe
face_mesh = mp.solutions.face_mesh
mesh = face_mesh.FaceMesh(min_detection_confidence=0.5,
                          min_tracking_confidence=0.5)

# Constants
left_eye_lm = [33, 246, 161, 160, 159, 158, 157,
               173, 133, 155, 154, 153, 145, 144, 163, 7]

right_eye_lm = [263, 466, 388, 387, 386, 385, 384,
                398, 362, 382, 381, 380, 374, 373, 390, 249]

# Open video stream
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    height, width, _ = frame.shape
    frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)

    results = mesh.process(frame)

    # pm = PupilTracker(frame)
    # left_eye, right_eye = pm.eye_coords(
    #     pm.left_eye_lm), pm.eye_coords(pm.right_eye_lm)

    # l_mask = pm.get_mask(left_eye)
    # r_mask = pm.get_mask(right_eye)

    # l_origin, l_roi = pm.eye_roi(
    #     left_eye, l_mask)
    # r_origin, r_roi = pm.eye_roi(
    #     right_eye, r_mask)

    # l_gray = cv2.cvtColor(l_roi, cv2.COLOR_RGB2GRAY)
    # r_gray = cv2.cvtColor(r_roi, (cv2.COLOR_RGB2GRAY))

    # l_pupil_x, l_pupil_y = pm.get_centroid(l_gray, l_origin, draw=False)
    # r_pupil_x, r_pupil_y = pm.get_centroid(r_gray, r_origin, draw=False)

    # Left and right point for width in left eye: [33, 133]
    # Upper and lower: [159, 145]
    # Left and right point for width in right eye: [362, 263]
    # Upper and lower: [386, 374]

    # Calculate eye_center
    top_left = int(results.multi_face_landmarks[0].landmark[159].x *
                   width), int(results.multi_face_landmarks[0].landmark[159].y * height)
    bottom_left = int(results.multi_face_landmarks[0].landmark[
        145].x * width), int(results.multi_face_landmarks[0].landmark[145].y * height)
    outer_left = int(results.multi_face_landmarks[0].landmark[
        33].x * width), int(results.multi_face_landmarks[0].landmark[33].y * height)
    inner_left = int(results.multi_face_landmarks[0].landmark[
        133].x * width), int(results.multi_face_landmarks[0].landmark[133].y * height)

    def line(p1, p2):
        A = (p1[1] - p2[1])
        B = (p2[0] - p1[0])
        C = (p1[0]*p2[1] - p2[0]*p1[1])
        return A, B, -C

    def intersection(L1, L2):
        D = L1[0] * L2[1] - L1[1] * L2[0]
        Dx = L1[2] * L2[1] - L1[1] * L2[2]
        Dy = L1[0] * L2[2] - L1[2] * L2[0]
        if D != 0:
            x = Dx / D
            y = Dy / D
            return int(x), int(y)
        else:
            return False

    L1 = line(top_left, bottom_left)
    L2 = line(outer_left, inner_left)

    R = intersection(L1, L2)

    # cv2.line(frame, top_left, bottom_left, (0, 0, 255), 1)
    # cv2.line(frame, outer_left, inner_left, (0, 0, 255), 1)
    cv2.circle(frame, R, 2, (0, 0, 255), -1)

    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.imshow("webcam feed", frame)

    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
