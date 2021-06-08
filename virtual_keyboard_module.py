# import libraires
import cv2


class VirtualKeyboard():
    def __init__(self):
        self.key_list = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0",
                         "Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P",
                         "A", "S", "D", "F", "G", "H", "J", "K", "L", "?",
                         "Z", "X", "C", "V", "B", "N", "M", ",", ".", "!"]

    def get_keys_shape(self, height, width):
        self.key_height = height//4
        self.key_width = width//10
        return self.key_height, self.key_width

    def draw_keyboard(self, frame):
        it_counter = 0
        height, width, _ = frame.shape
        self.key_height = height//4
        self.key_width = width//10
        for columns in range(0, 4):
            for rows in range(0, 10):
                cv2.rectangle(frame, (rows * self.key_width, (columns*self.key_height)),
                              ((rows + 1)*self.key_width, (columns+1)*self.key_height), (0, 255, 0), 1)
                cv2.putText(frame, self.key_list[it_counter], (((self.key_width - (self.key_width // 2)-(self.key_width // 3)) + (self.key_width*rows)),
                                                               ((self.key_height // 2) + (self.key_height // 4) + self.key_height*columns)),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
                it_counter += 1
        return frame


# def main():
#     cap = cv2.VideoCapture(1)
#     keyboard = VirtualKeyboard()
#     while cap.isOpened():
#         ret, frame = cap.read()

#         frame = keyboard.draw_keyboard(frame)

#         cv2.imshow("webcam", frame)

#         if cv2.waitKey(20) & 0xFF == 27:
#             break


# if __name__ == "__main__":
#     main()
