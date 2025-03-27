import fiftyone as fo
import fiftyone.zoo as foz

dataset = foz.load_zoo_dataset(
    "open-images-v7",
    split="train",  #  "train", "validation" lub "test"
    label_types=["detections"],
    classes=["Bird"],# "Balloon", "Aircraft", "Airplane", "Kite"],"Helicopter"
    max_samples=2000
)

# Eksport pobranych obrazów na dysk
export_dir = "/bird"
dataset.export(
    export_dir=export_dir,
    dataset_type=fo.types.YOLOv5Dataset,
    label_field="ground_truth"
)

