import cv2
from ultralytics import YOLO

model = YOLO('detection/last.pt')

def process_video(source):
    cap = cv2.VideoCapture(source)

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # Kodek wideo
    out = cv2.VideoWriter('detection/output4.mp4', fourcc, fps, (frame_width, frame_height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)

        for r in results:
            boxes = r.boxes
            for box in boxes:
                if box.conf > 0.5:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame,
                              f'{model.names[int(box.cls)]} {box.conf[0]:.2f}',
                              (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                              0.9, (0, 255, 0), 2)

        out.write(frame)
        # cv2.imshow('Object Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

# source = 0  # 0 camera or path 'path/to/video.mp4'
# source = 'detection/Airshow2024.mp4'
# source = 'detection/BalloonFiesta.mp4'
# source = 'detection/DronevsBird.mp4'
source = 'detection/Eagleattacks.mp4'

process_video(source)
