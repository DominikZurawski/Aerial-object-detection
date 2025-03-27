import os
import glob
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

images_folder = "data/test/drone"
masks_folder = "data/test/masks"
output_folder = "data/test/drone"

os.makedirs(output_folder, exist_ok=True)

# Wartości pikseli reprezentujące drony w Twoich maskach
DRONE_PIXELS = [204, 205, 206, 207, 208]

def extract_bounding_boxes(mask_array):
    """Wyodrębnia bboxy dla zdefiniowanych wartości pikseli."""
    # Tworzymy maskę logiczną dla pikseli drona
    drone_mask = np.isin(mask_array, DRONE_PIXELS)

    if not np.any(drone_mask):
        print("[DEBUG] Brak pikseli drona w masce")
        return []

    # Etykietowanie komponentów
    from scipy import ndimage
    labeled, num_features = ndimage.label(drone_mask)

    boxes = []
    for i in range(1, num_features + 1):
        y, x = np.where(labeled == i)
        x_min, x_max = np.min(x), np.max(x)
        y_min, y_max = np.min(y), np.max(y)
        boxes.append((x_min, y_min, x_max, y_max))

    return boxes

# Funkcja do wizualnej weryfikacji
def debug_plot(image, mask, boxes):
    plt.figure(figsize=(15, 5))

    plt.subplot(131)
    plt.imshow(image)
    plt.title("Oryginalny obraz")

    plt.subplot(132)
    plt.imshow(mask, cmap='gray')
    plt.title("Maska")

    plt.subplot(133)
    plt.imshow(image)
    for (x1, y1, x2, y2) in boxes:
        plt.plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], 'r-')
    plt.title("Bounding boxy")

    plt.tight_layout()
    plt.show()

for image_file in os.listdir(images_folder):
    if not image_file.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue

    print(f"\n--- Przetwarzanie: {image_file} ---")

    try:
        # Wczytaj obraz
        img_path = os.path.join(images_folder, image_file)
        img = Image.open(img_path)

        # Znajdź odpowiadającą maskę
        base_number = image_file.replace("Image", "").split(".")[0]
        mask_pattern = os.path.join(masks_folder, f"Mask{base_number}.png_*")
        masks = glob.glob(mask_pattern)

        if not masks:
            print(f"[ERROR] Brak maski dla {image_file}")
            continue

        mask_path = masks[0]
        mask = Image.open(mask_path).convert("L")
        mask_array = np.array(mask)

        # Wyodrębnij bboxy
        boxes = extract_bounding_boxes(mask_array)

        # Weryfikacja wizualna
        # debug_plot(img, mask_array, boxes)

        # Zapisz do pliku YOLO
        output_path = os.path.join(output_folder, os.path.splitext(image_file)[0] + ".txt")
        with open(output_path, 'w') as f:
            for x1, y1, x2, y2 in boxes:
                width = img.width
                height = img.height
                x_center = ((x1 + x2) / 2) / width
                y_center = ((y1 + y2) / 2) / height
                w = (x2 - x1) / width
                h = (y2 - y1) / height
                f.write(f"3 {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")

        print(f"Zapisano {len(boxes)} bboxów")

    except Exception as e:
        print(f"Błąd: {str(e)}")
        continue
