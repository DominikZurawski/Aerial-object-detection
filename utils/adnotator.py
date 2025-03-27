import os
import cv2
import numpy as np
import torch
from ultralytics.data.annotator import auto_annotate
from ultralytics.utils.ops import segments2boxes

# Katalogi źródłowe i docelowe
base_dir = 'C:/Users/Dominik/Downloads/other/data/test'
output_dir = 'auto_labels'
converted_output_dir = 'converted_labels'
classes = ['airplane', 'balloon', 'bird', 'drone', 'helicopter', 'sky']

# Uruchomienie auto_annotate i natychmiastowa konwersja adnotacji
def annotate_and_convert(base_dir, output_dir, converted_output_dir, classes):
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(converted_output_dir, exist_ok=True)

    for class_name in classes:
        class_dir = os.path.join(base_dir, class_name)
        class_output_dir = os.path.join(output_dir, class_name)
        converted_class_output_dir = os.path.join(converted_output_dir, class_name)
        os.makedirs(class_output_dir, exist_ok=True)
        os.makedirs(converted_class_output_dir, exist_ok=True)

        auto_annotate(
            data=class_dir,
            det_model='yolov8n.pt',
            sam_model='mobile_sam.pt',
            output_dir=class_output_dir
        )

        # Przechodzimy od razu do konwersji adnotacji
        for txt_file in os.listdir(class_output_dir):
            if txt_file.endswith('.txt'):
                txt_path = os.path.join(class_output_dir, txt_file)
                converted_txt_path = os.path.join(converted_class_output_dir, txt_file)
                image_path = os.path.join(base_dir, class_name, os.path.splitext(txt_file)[0])

                # Szukamy odpowiadającego obrazu
                for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
                    if os.path.exists(image_path + ext):
                        image_path += ext
                        break
                else:
                    continue  # Jeśli obraz nie istnieje, pomijamy plik

                # Pobranie wymiarów obrazu
                img = cv2.imread(image_path)
                if img is None:
                    continue
                height, width, _ = img.shape

                new_lines = []
                with open(txt_path, 'r') as f:
                    lines = f.readlines()

                for line in lines:
                    values = line.strip().split()
                    if len(values) < 5:
                        continue  # Pomijamy błędne wiersze

                    class_id = int(values[0])
                    coords = list(map(float, values[1:]))

                    if len(coords) > 4:
                        # Konwersja segmentacji na bounding box
                        segment = np.array(coords, dtype=np.float32).reshape(-1, 2)
                        segment[:, 0] *= width
                        segment[:, 1] *= height
                        segment = torch.tensor(segment, dtype=torch.float32)
                        bbox = segments2boxes(segment).flatten()
                        x_min, y_min, x_max, y_max = bbox[:4]
                    else:
                        # Jeśli mamy dokładnie 4 wartości, używamy ich bez zmian
                        x_min, y_min, x_max, y_max = coords[:4]

                    # Konwersja do formatu YOLO
                    x_center = (x_min + x_max) / 2 / width
                    y_center = (y_min + y_max) / 2 / height
                    bbox_width = (x_max - x_min) / width
                    bbox_height = (y_max - y_min) / height

                    new_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}\n")

                # Zapisujemy plik w nowym katalogu
                with open(converted_txt_path, 'w') as f:
                    f.writelines(new_lines)

# Uruchomienie procesu
annotate_and_convert(base_dir, output_dir, converted_output_dir, classes)
print("✅ Adnotacje i konwersja do formatu YOLO zakończone.")
# następnym krokiem powinno być przejrzenie obiektów w plikach txt