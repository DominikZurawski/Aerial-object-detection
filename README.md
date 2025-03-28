# Aerial Object Detection

## 📌 Project Description
This project focuses on aerial object detection using YOLOv8 and YOLOv11 models. The models were trained to recognize the following object classes:
- ✈️ `airplane`
- 🎈 `balloon`
- 🐦 `bird`
- 🚁 `drone`
- 🚁 `helicopter`
- ☁️ `sky`

## 📊 Training Data
The dataset consisted of:
- **800** training images per class
- **200** validation images per class
- **250** test images per class

### 📂 Data Sources
The images were sourced from the following datasets:
- [ManzoorUmair Cloud Cover Dataset (MUCCD)](https://www.kaggle.com/datasets/umairatwork/manzoorumair-cloud-cover-dataset-muccd)
- Open Images V7
- [Cranfield Synthetic Drone Detection Dataset](https://huggingface.co/datasets/mazqtpopx/cranfield-synthetic-drone-detection)

## 🚀 Model Training
Training was conducted on YOLOv8 and YOLOv11 models. Comparison results:

| Model        | Parameters (M) | GFLOPs | Size (MB) | mAP50 | mAP50-95 | Precision | Recall |
|-------------|--------------|--------|-------------|--------|----------|-----------|--------|
| YOLOv8_exp1 | 3.012        | 8.2    | 5.95        | 0.730  | 0.496    | 0.824     | 0.689  |
| YOLOv8_exp3 | 3.012        | 8.2    | 5.96        | 0.725  | 0.493    | 0.822     | 0.690  |
| YOLOv11_exp2 | 2.591        | 6.4    | 5.20        | 0.726  | 0.487    | 0.801     | 0.686  |

🛠 **Best model:** `YOLOv8_exp1` (mAP50-95 = 0.496, precision = 0.824, recall = 0.689)

## 📈 Results Visualization
- The trained model has been applied to a video, and the output is available at: `/detection/output.mp4`



