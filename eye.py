# import libraries
import cv2
import mediapipe as mp
import numpy as np

# instantiate mediapipe
face_mesh = mp.solutions.face_mesh

# constants
left_eye_lm = [33, 246, 161, 160, 159, 158, 157,
               173, 133, 155, 154, 153, 145, 144, 163, 7]

right_eye_lm = [263, 466, 388, 387, 386, 385, 384,
                398, 362, 382, 381, 380, 374, 373, 390, 249]

# functions


def eye_coords(frame, landmarks):
    height, width, _ = frame.shape
    eye_x = []
    eye_y = []
    for lm in landmarks:
        # mp return normalized coords, so we multiply
        eye_x.append(
            int((results.multi_face_landmarks[0].landmark[lm].x) * width))
        eye_y.append(
            int((results.multi_face_landmarks[0].landmark[lm].y) * height))
    eye = list(zip(eye_x, eye_y))
    eye = np.array(eye, dtype="int")
    return eye


# video capture
cap = cv2.VideoCapture(0)
with face_mesh.FaceMesh(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as mesh:
    while cap.isOpened():
        ret, frame = cap.read()
        height, width, _ = frame.shape
        frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
        results = mesh.process(frame)
        try:
            if results:
                face_landmark = results.multi_face_landmarks

                if len(face_landmark) != 0:
                    left_eye = eye_coords(frame, left_eye_lm)
                    right_eye = eye_coords(frame, right_eye_lm)

            # mask
            mask = np.zeros((height, width), np.uint8)
            cv2.polylines(mask, [left_eye], True, 255, 2)
            cv2.fillPoly(mask, [left_eye], 255)
            eyes = cv2.bitwise_and(frame, frame, mask=mask)

            # Cropping on the eye
            l_min_x = np.min(left_eye[:, 0])
            l_max_x = np.max(left_eye[:, 0])
            l_min_y = np.min(left_eye[:, 1])
            l_max_y = np.max(left_eye[:, 1])
            origin = (l_min_x, l_min_y)
            print(l_min_x, l_max_x, l_min_y, l_max_y)
            left_eye_frame = eyes.copy()[l_min_y:l_max_y, l_min_x:l_max_x]
            l_eye_gray = cv2.cvtColor(left_eye_frame, cv2.COLOR_RGB2GRAY)

            _, l_thresh = cv2.threshold(
                cv2.cvtColor(left_eye_frame, cv2.COLOR_BGR2GRAY), 50, 255, cv2.THRESH_BINARY_INV)

            contours, _ = cv2.findContours(
                l_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            contours = sorted(
                contours, key=lambda x: cv2.contourArea(x), reverse=True)

            for cnt in contours:
                cv2.drawContours(left_eye_frame, [cnt], -1, (0, 255, 0), 1)
                moments = cv2.moments(cnt)
                centroid_x, centroid_y = int(
                    moments["m10"]/moments["m00"]), int(moments["m01"]/moments["m00"])
                cv2.circle(left_eye_frame, (centroid_x,
                                            centroid_y), 2, (0, 255, 0))

                break

            pupil_x, pupil_y = origin[0] + centroid_x, origin[1] + centroid_y

            cv2.line(frame, (pupil_x - 5, pupil_y),
                     (pupil_x + 5, pupil_y), (0, 255, 0), 1)
            cv2.line(frame, (pupil_x, pupil_y - 5),
                     (pupil_x, pupil_y + 5), (0, 255, 0), 1)
        except:
            pass
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imshow("masked frame", eyes)
        cv2.imshow("webcam feed", frame)
        cv2.imshow("left_eye_gray", cv2.cvtColor(
            l_eye_gray, cv2.COLOR_RGB2BGR))
        cv2.imshow("thresh", l_thresh)

        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
