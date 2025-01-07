import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

model = load_model('model.keras')

def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open video capture.")
        return

    text = ""
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 2
    font_color = (2, 2, 255)
    thickness = 2
    position = np.array([300, 250])
    w, h = 100, 100
    show = None
    while True:
        if show:
            text = show
        ret, frame = cap.read()
        frame_height, frame_width, _ = frame.shape
        key = cv2.waitKey(1) & 0xFF
        if not ret:
            print("Error:")
            break
        x1, y1 = top_left = (frame_width // 2 - w // 2, frame_height // 2 - h // 2)
        x2, y2 = bottom_right = (frame_width // 2 + w // 2, frame_height // 2 + h // 2)
        cv2.rectangle(frame, top_left, bottom_right, (0, 0, 255), 2)
        cv2.putText(frame, text, position, font, font_scale, font_color, thickness, cv2.LINE_AA)
        cv2.imshow('Video Feed', frame)

        if 1:
            fl = frame[y1:y2, x1:x2]
            gray = cv2.cvtColor(fl, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            resized_digit = cv2.resize(binary, (28, 28), interpolation=cv2.INTER_AREA)
            normalized_digit = resized_digit / 255.0
            input_data = normalized_digit.reshape(1, 28, 28)
            prediction = model.predict(input_data)
            predicted_digit = np.argmax(prediction)
            predicted_probability = prediction[0][predicted_digit]
            if predicted_probability > 0.7:
                show = str(np.argmax(prediction))

        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(fl.shape)

main()
