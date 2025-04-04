import os
import argparse
import cv2
import torch
import torchvision
from PIL import Image
from torchvision import transforms

def parse_args():
    parser = argparse.ArgumentParser(description='Object detection in video using the EffNetB2 model')
    parser.add_argument('--source', type=str, default='0', help='Video source. 0 for camera, path for video file')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the EffNetB2 model')
    parser.add_argument('--output', type=str, default='output.mp4', help='Path to save the resulting video')
    parser.add_argument('--skip_frames', type=int, default=1, help='Number of frames to be skipped between processed')
    parser.add_argument('--conf_threshold', type=float, default=0.5, help='Detection confidence threshold')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Calculation device (cuda/cpu)')
    parser.add_argument('--class_names', type=str, nargs='+', default=["airplane", "balloon", "bird", "drone", "helicopter", "sky"], help='Nazwy klas')
    parser.add_argument('--num_classes', type=int, default=6, help='Number of classes')
    return parser.parse_args()

def load_model(model_path, device, num_classes):
    model = torchvision.models.efficientnet_b2(pretrained=False)
    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(p=0.3, inplace=True),
        torch.nn.Linear(in_features=1408, out_features=num_classes)
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def get_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def process_frame(frame, model, transform, device, class_names, conf_threshold):
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    input_tensor = transform(pil_image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probs, 1)

    if confidence.item() >= conf_threshold:
        label = f"{class_names[predicted.item()]}: {confidence.item():.2f}"
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return frame, predicted.item(), confidence.item()

def process_video(source, model, transform, output_path, skip_frames, conf_threshold, device, class_names):
    if source.isdigit():
        cap = cv2.VideoCapture(int(source))
    else:
        cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print(f"Video source cannot be opened: {source}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    frame_count = 0
    last_prediction = None
    last_confidence = 0.0

    print(f"Video processing from the source has begun: {source}")
    print(f"Output file: {output_path}")
    print(f"Classification every {skip_frames} frames")

    try:
        while cap.isOpened():
            if (frame_count % skip_frames != 0) and frame_count > 0:
                cap.grab()
                frame_count += 1
                continue

            ret, frame = cap.read()

            if not ret:
                break

            frame_count += 1

            frame, pred, conf = process_frame(frame, model, transform, device, class_names, conf_threshold)

            if conf >= conf_threshold:
                last_prediction = pred
                last_confidence = conf

            if last_prediction is not None:
                info_text = f"Last detection: {class_names[last_prediction]} ({last_confidence:.2f})"
                cv2.putText(frame, info_text, (10, frame_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            out.write(frame)

            cv2.imshow('Object detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            if frame_count % 100 == 0:
                print(f"Processed {frame_count} frames")

    except KeyboardInterrupt:
        print("User processing interrupted.")
    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print(f"Video processing completed. Processed {frame_count} frames.")

def main():
    args = parse_args()

    device = torch.device(args.device)
    model = load_model(args.model_path, device, args.num_classes)
    transform = get_transforms()

    process_video(
        args.source,
        model,
        transform,
        args.output,
        args.skip_frames,
        args.conf_threshold,
        device,
        args.class_names
    )

if __name__ == "__main__":
    main()
