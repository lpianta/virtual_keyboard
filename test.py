import cv2
import virtual_keyboard_module as vmk
from pupil_module import PupilTracker

keyboard = vmk.VirtualKeyboard()
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    try:
        pm = PupilTracker(frame)
        left_eye, right_eye = pm.eye_coords(
            pm.left_eye_lm), pm.eye_coords(pm.right_eye_lm)

        l_mask = pm.get_mask(left_eye)
        r_mask = pm.get_mask(right_eye)

        l_origin, l_roi = pm.eye_roi(
            left_eye, l_mask)
        r_origin, r_roi = pm.eye_roi(
            right_eye, r_mask)

        l_gray = cv2.cvtColor(l_roi, cv2.COLOR_RGB2GRAY)
        r_gray = cv2.cvtColor(r_roi, (cv2.COLOR_RGB2GRAY))

        l_pupil_x, l_pupil_y = pm.get_centroid(l_gray, l_origin)
        r_pupil_x, r_pupil_y = pm.get_centroid(r_gray, r_origin)
    except:
        pass

    frame = keyboard.draw_keyboard(frame)
    cv2.imshow("webcam feed", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
