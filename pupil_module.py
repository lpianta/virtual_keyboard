# import libraries
import cv2
import mediapipe as mp
import numpy as np


class PupilTracker():
    def __init__(self, frame):
        # constants
        self.left_eye_lm = [33, 246, 161, 160, 159, 158, 157,
                            173, 133, 155, 154, 153, 145, 144, 163, 7]

        self.right_eye_lm = [263, 466, 388, 387, 386, 385, 384,
                             398, 362, 382, 381, 380, 374, 373, 390, 249]

        # shape
        self.frame = frame
        self.height, self.width, _ = self.frame.shape

        # instantiate mediapipe
        self.face_mesh = mp.solutions.face_mesh
        self.mesh = self.face_mesh.FaceMesh(
            min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.results = self.mesh.process(self.frame)

    def eye_coords(self, landmarks):
        eye_x = [int((self.results.multi_face_landmarks[0].landmark[lm].x)
                     * self.width) for lm in landmarks]
        eye_y = [int((self.results.multi_face_landmarks[0].landmark[lm].y)
                     * self.height) for lm in landmarks]
        self.eye = np.array(list(zip(eye_x, eye_y)), dtype="int")
        return self.eye

    def get_mask(self, eye):
        mask = np.zeros((self.height, self.width), np.uint8)
        cv2.polylines(mask, [eye], True, 255, 2)
        cv2.fillPoly(mask, [eye], 255)
        self.eyes = cv2.bitwise_and(self.frame, self.frame, mask=mask)
        return self.eyes

    def eye_roi(self, eye, mask):
        min_x = np.min(eye[:, 0])
        max_x = np.max(eye[:, 0])
        min_y = np.min(eye[:, 1])
        max_y = np.max(eye[:, 1])
        self.origin = (min_x, min_y)
        self.eye_frame = mask.copy()[min_y:max_y, min_x:max_x]
        return self.origin, self.eye_frame

    def get_centroid(self, gray_roi, origin, draw=True):
        _, thresh = cv2.threshold(gray_roi, 100, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(
            contours, key=lambda x: cv2.contourArea(x), reverse=True)
        for cnt in contours:
            moments = cv2.moments(cnt)
        try:
            centroid_x, centroid_y = int(
                moments["m10"]/moments["m00"]), int(moments["m01"]/moments["m00"])

            pupil_x, pupil_y = pupil_x, pupil_y = origin[0] + \
                centroid_x, origin[1] + centroid_y
            if draw:
                cv2.line(self.frame, (pupil_x - 5, pupil_y),
                         (pupil_x + 5, pupil_y), (0, 255, 0), 1)
                cv2.line(self.frame, (pupil_x, pupil_y - 5),
                         (pupil_x, pupil_y + 5), (0, 255, 0), 1)
            return pupil_x, pupil_y
        except ZeroDivisionError:
            pass


# # video capture
# cap = cv2.VideoCapture(0)
# while cap.isOpened():
#     ret, frame = cap.read()
#     frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
#     pupil_tracker = PupilTracker(frame)

#     left_eye, right_eye = pupil_tracker.eye_coords(
#         pupil_tracker.left_eye_lm), pupil_tracker.eye_coords(pupil_tracker.right_eye_lm)

#     l_mask = pupil_tracker.get_mask(left_eye)
#     r_mask = pupil_tracker.get_mask(right_eye)

#     l_origin, l_roi = pupil_tracker.eye_roi(
#         left_eye, l_mask)
#     r_origin, r_roi = pupil_tracker.eye_roi(
#         right_eye, r_mask)

#     l_gray = cv2.cvtColor(l_roi, cv2.COLOR_RGB2GRAY)
#     r_gray = cv2.cvtColor(r_roi, (cv2.COLOR_RGB2GRAY))
#     try:
#         l_pupil_x, l_pupil_y = pupil_tracker.get_centroid(l_gray, l_origin)
#         r_pupil_x, r_pupil_y = pupil_tracker.get_centroid(r_gray, r_origin)
#     except:
#         pass

#     frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

#     #cv2.imshow("masked frame", eyes)
#     cv2.imshow("webcam feed", frame)
#     # cv2.imshow("left_eye_gray", cv2.cvtColor(
#     #    l_eye_gray, cv2.COLOR_RGB2BGR))
#     #cv2.imshow("thresh", l_thresh)

#     if cv2.waitKey(20) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()
