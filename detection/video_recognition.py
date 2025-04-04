import torch
import cv2
import numpy as np
from torchvision import models, transforms
from PIL import Image
import time
import os
import argparse

def load_model_and_transforms(model_path: str,
                             num_classes: int = 6,
                             device: str = "cuda" if torch.cuda.is_available() else "cpu"):
    vit_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    model = models.vit_b_16(weights=None)

    for param in model.parameters():
        param.requires_grad = False

    model.heads = torch.nn.Sequential(
        torch.nn.Linear(768, num_classes)
    )

    model.load_state_dict(torch.load(model_path, map_location=device))
    return model, vit_transforms

def process_video(model: torch.nn.Module,
                 transforms: transforms.Compose,
                 class_names: list,
                 input_source: str = "0",
                 output_path: str = None,
                 show_preview: bool = True,
                 skip_frames: int = 0):
    """
    Process video stream with real-time classification.
    """
    device = next(model.parameters()).device
    cap = cv2.VideoCapture(int(input_source) if input_source.isdigit() else input_source)

    if not cap.isOpened():
        raise ValueError(f"Could not open video source: {input_source}")

    writer = None
    if output_path:
        output_size = (224, 224)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 30
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        writer = cv2.VideoWriter(output_path, fourcc, fps, output_size)

        if not writer.isOpened():
            raise RuntimeError("Failed to initialize video writer")

    class_colors = {
        "airplane": (0, 255, 0),      # Green
        "balloon": (0, 165, 255),     # Orange
        "bird": (255, 255, 0),        # Cyan
        "drone": (0, 0, 255),         # Red
        "helicopter": (255, 0, 255),  # Magenta
        "sky": (255, 255, 255)        # White
    }

    frame_counter = 0
    print("Processing started. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if skip_frames > 0 and frame_counter % (skip_frames + 1) != 0:
            frame_counter += 1
            continue

        # Preprocessing
        frame = cv2.resize(frame, (224, 224))
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)

        with torch.no_grad():
            start_time = time.time()
            input_tensor = transforms(pil_image).unsqueeze(0).to(device)
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            conf, cls_idx = torch.max(probabilities, 0)
            inference_time = time.time() - start_time

        # Get class info
        class_name = class_names[cls_idx.item()]
        confidence = conf.item()
        color = class_colors.get(class_name, (255, 255, 255))

        # Draw overlay
        label = f"{class_name}: {confidence:.2f}"
        cv2.putText(frame, label, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(frame, f"Inference: {inference_time*1000:.1f}ms",
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

        # Write to output file
        if writer:
            writer.write(frame)

        # Show preview
        if show_preview:
            cv2.imshow('ViT Object Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        frame_counter += 1

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ViT-based Video Object Detection')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model file (.pth)')
    parser.add_argument('--input', type=str, default='0',
                       help='Input source (0 for camera, or path to video file)')
    parser.add_argument('--output', type=str,
                       help='Output video path (optional)')
    parser.add_argument('--no-preview', action='store_false',
                       help='Disable live preview window')
    parser.add_argument('--skip-frames', type=int, default=0,
                      help='Number of frames to skip between processed frames (reduces processing load)')

    args = parser.parse_args()

    # Class mapping
    CLASS_NAMES = [
        "airplane",
        "balloon",
        "bird",
        "drone",
        "helicopter",
        "sky"
    ]

    if not os.path.exists(args.model):
        raise FileNotFoundError(f"Model file not found: {args.model}")

    print("Loading model...")
    model, transforms = load_model_and_transforms(args.model)
    print(f"Model loaded successfully. Using device: {next(model.parameters()).device}")

    print("Starting video processing...")
    process_video(
        model=model,
        transforms=transforms,
        class_names=CLASS_NAMES,
        input_source=args.input,
        output_path=args.output,
        show_preview=args.no_preview,
        skip_frames=args.skip_frames
    )
