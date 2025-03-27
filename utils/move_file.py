import os
import shutil
import random

def distribute_files(base_folder):
    categories = ["airplane", "drone", "sky", "helicopter", "bird", "balloon"]
    output_txt_folder = os.path.join(base_folder, "output_txt")
    images_train_folder = os.path.join(base_folder, "images", "train")
    images_val_folder = os.path.join(base_folder, "images", "val")
    labels_train_folder = os.path.join(base_folder, "labels", "train")
    labels_val_folder = os.path.join(base_folder, "labels", "val")

    os.makedirs(images_train_folder, exist_ok=True)
    os.makedirs(images_val_folder, exist_ok=True)
    os.makedirs(labels_train_folder, exist_ok=True)
    os.makedirs(labels_val_folder, exist_ok=True)

    valid_extensions = (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG")

    total_train = 0
    total_val = 0

    for category in categories:
        category_folder = os.path.join(base_folder, category)
        images = [f for f in os.listdir(category_folder) if f.endswith(valid_extensions)]
        random.shuffle(images)

        train_images = images[:800] if len(images) >= 800 else images[:int(len(images) * 0.8)]
        val_images = images[800:1000] if len(images) >= 1000 else images[int(len(images) * 0.8):]

        for img in train_images:
            shutil.copy(os.path.join(category_folder, img), os.path.join(images_train_folder, img))
            total_train += 1
            txt_file = os.path.splitext(img)[0] + ".txt"
            if os.path.exists(os.path.join(output_txt_folder, txt_file)):
                shutil.copy(os.path.join(output_txt_folder, txt_file), os.path.join(labels_train_folder, txt_file))

        for img in val_images:
            shutil.copy(os.path.join(category_folder, img), os.path.join(images_val_folder, img))
            total_val += 1
            txt_file = os.path.splitext(img)[0] + ".txt"
            if os.path.exists(os.path.join(output_txt_folder, txt_file)):
                shutil.copy(os.path.join(output_txt_folder, txt_file), os.path.join(labels_val_folder, txt_file))

        print(f"Przetworzono kategorię: {category} - Train: {len(train_images)}, Val: {len(val_images)}")

    print(f"\nPodsumowanie: Train = {total_train}, Val = {total_val}")

base_folder_path = "C:/Users/Dominik/Downloads/other/data/data — kopia"
distribute_files(base_folder_path)
