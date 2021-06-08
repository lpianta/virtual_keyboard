# import libraires
import cv2

# costants
key_list = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0",
            "Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P",
            "A", "S", "D", "F", "G", "H", "columns", "K", "L", "?",
            "Z", "X", "C", "V", "B", "N", "M", ",", ".", "!"]

# functions


def get_key_values(height, width):
    return height // 4, width // 10


# video stream
cap = cv2.VideoCapture(1)

while cap.isOpened():
    ret, frame = cap.read()
    height, width, _ = frame.shape
    key_height, key_width = get_key_values(height, width)
    it_counter = 0
    for columns in range(0, 4):
        for rows in range(0, 10):
            cv2.rectangle(frame, (rows * key_width, (columns*key_height)),
                          ((rows + 1)*key_width, (columns+1)*key_height), (0, 255, 0), 1)
            cv2.putText(frame, key_list[it_counter], (((key_width - (key_width // 2)-(key_width // 3)) + (key_width*rows)), ((key_height // 2) + (key_height // 4) + key_height*columns)),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
            it_counter += 1

    cv2.imshow("webcam", frame)

    if cv2.waitKey(20) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
