import os

def create_txt_files(image_folder):
    for filename in os.listdir(image_folder):
        if filename.lower().endswith(".jpg"):
            txt_filename = os.path.splitext(filename)[0] + ".txt"
            txt_filepath = os.path.join(image_folder, txt_filename)

            with open(txt_filepath, "w") as txt_file:
                txt_file.write("5 0.5 0.5 1.0 1.0\n")

            print(f"Utworzono: {txt_filepath}")

# Przykład użycia
folder_path = "C:/Users/Dominik/Downloads/other/data/test/sky"
create_txt_files(folder_path)