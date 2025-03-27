import cv2
from ultralytics import YOLO

# Inicjalizacja modelu YOLOv8
model = YOLO('last.pt')  # Ścieżka do Twojego wytrenowanego modelu

# Funkcja do przetwarzania wideo
def process_video(source):
    cap = cv2.VideoCapture(source)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detekcja obiektów
        results = model(frame)

        # Rysowanie wyników na klatce
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                conf = box.conf[0]
                cls = int(box.cls[0])

                if conf > 0.5:  # Próg pewności
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    cv2.putText(frame, f'{model.names[cls]} {conf:.2f}',
                                (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX,
                                0.9, (0, 255, 0), 2)

        cv2.imshow('Object Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Wybór źródła wideo
# source = 0  # 0 dla kamery, lub ścieżka do pliku wideo, np. 'path/to/video.mp4'
# source = 'DronevsBird.mp4'
# source = 'Eagleattacks.mp4'
source = 'Airshow2024.mp4'
# source = 'BalloonFiesta.mp4'
process_video(source)
