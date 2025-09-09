#!/usr/bin/env python3

import os
import argparse
import torch
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from models.csrnet import csrnet
from models.density_classifier import density_classifier
from models.yolo_counter import YOLOCounter


def load_models(models_dir, device):
    """
    Load trained models
    """
    print("Loading models...")
    
    try:
        # Load CSRNet
        csrnet_model = csrnet(pretrained=False).to(device)
        csrnet_model.load_state_dict(torch.load(
            os.path.join(models_dir, 'csrnet_best.pth'),
            map_location=device
        ))
        csrnet_model.eval()
        print("CSRNet model loaded successfully")
        
        # Load density classifier
        classifier_model = density_classifier(pretrained=False).to(device)
        classifier_model.load_state_dict(torch.load(
            os.path.join(models_dir, 'classifier_best.pth'),
            map_location=device
        ))
        classifier_model.eval()
        print("Classifier model loaded successfully")
        
        # Use pretrained YOLOv8n model (avoid torch.hub)
        # This will use the model we downloaded during training
        print("Using pretrained YOLOv8n model")
        yolo_counter = YOLOCounter(model_size='n', pretrained=True)
        
        return csrnet_model, classifier_model, yolo_counter
        
    except Exception as e:
        print(f"Error loading models: {e}")
        raise


def preprocess_image(image_path, target_size=384):
    """
    Load and preprocess an image for the hybrid crowd counter
    """
    # Load image
    image = Image.open(image_path).convert('RGB')
    
    # Preprocessing for classifier and CSRNet
    transform = transforms.Compose([
        transforms.Resize((target_size, target_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    tensor = transform(image)
    
    return image, tensor.unsqueeze(0)  # Add batch dimension


def count_crowd(image_path, csrnet_model, classifier_model, yolo_counter, device, output_dir=None):
    """
    Count crowd in image using hybrid model approach
    """
    # Load and preprocess image
    original_image, img_tensor = preprocess_image(image_path)
    img_tensor = img_tensor.to(device)
    
    with torch.no_grad():
        # Classify density
        density_pred = classifier_model(img_tensor)
        is_dense = (density_pred > 0.5).item()
        
        if is_dense:
            print("Classified as DENSE crowd")
            # Use CSRNet for dense crowd
            density_map = csrnet_model(img_tensor)
            
            # Resize density map to original size if needed
            if density_map.size(2) != img_tensor.size(2) or density_map.size(3) != img_tensor.size(3):
                density_map = torch.nn.functional.interpolate(
                    density_map,
                    size=(img_tensor.size(2), img_tensor.size(3)),
                    mode='bilinear',
                    align_corners=False
                )
            
            # Calculate count from density map
            # Apply a scaling factor to calibrate the count
            # CSRNet tends to underestimate crowd counts, so we apply a multiplier
            scaling_factor = 8.0  # Adjust this based on validation results
            raw_count = density_map.sum().item()
            count = raw_count * scaling_factor
            
            # Create visualization
            if output_dir:
                # Convert density map for visualization
                density_map_np = density_map.squeeze().cpu().numpy()
                
                # Save density map as heatmap
                plt.figure(figsize=(10, 10))
                plt.subplot(1, 2, 1)
                plt.imshow(np.array(original_image))
                plt.title('Original Image')
                
                plt.subplot(1, 2, 2)
                plt.imshow(np.array(original_image))
                plt.imshow(density_map_np, alpha=0.6, cmap='jet')
                plt.title(f'Density Map (Raw: {raw_count:.1f}, Scaled: {count:.1f})')
                
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                    
                output_path = os.path.join(output_dir, os.path.basename(image_path).split('.')[0] + '_density.png')
                plt.savefig(output_path)
                plt.close()
                print(f"Visualization saved to {output_path}")
        else:
            print("Classified as SPARSE crowd")
            # Use YOLOv8 for sparse crowd
            # Convert from tensor to format expected by YOLO
            img_np = img_tensor.cpu().squeeze(0).permute(1, 2, 0).numpy()
            
            # Denormalize the image (convert from normalized to 0-255 range)
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img_np = img_np * std + mean
            img_np = (img_np * 255).astype(np.uint8)
            
            # Run detection
            counts, boxes_list = yolo_counter.count(img_np, device=device)
            count = counts[0] if counts else 0
            
            # Create visualization
            if output_dir:
                # Get original image for drawing
                img_draw = cv2.cvtColor(np.array(original_image), cv2.COLOR_RGB2BGR)
                
                # Draw bounding boxes
                boxes = boxes_list[0]
                for box in boxes:
                    # Convert normalized coordinates to pixel values
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    cv2.rectangle(img_draw, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)
                    
                output_path = os.path.join(output_dir, os.path.basename(image_path).split('.')[0] + '_detection.png')
                cv2.putText(img_draw, f"Person Count: {count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imwrite(output_path, img_draw)
                print(f"Visualization saved to {output_path}")
    
    print(f"Estimated crowd count: {count:.1f}")
    return count


def main(args):
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load models
    csrnet_model, classifier_model, yolo_counter = load_models(args.models_dir, device)
    
    # Create output directory if needed
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    
    if os.path.isfile(args.input):
        # Process single image
        count = count_crowd(
            args.input, 
            csrnet_model, 
            classifier_model, 
            yolo_counter, 
            device, 
            args.output_dir
        )
    else:
        # Process directory of images
        results = []
        for filename in sorted(os.listdir(args.input)):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(args.input, filename)
                print(f"Processing {filename}...")
                count = count_crowd(
                    image_path, 
                    csrnet_model, 
                    classifier_model, 
                    yolo_counter, 
                    device, 
                    args.output_dir
                )
                results.append((filename, count))
        
        # Save results to CSV
        if args.output_dir:
            import csv
            csv_path = os.path.join(args.output_dir, 'crowd_counts.csv')
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Image', 'Count'])
                writer.writerows(results)
            print(f"Results saved to {csv_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Count crowds in images using hybrid model')
    
    parser.add_argument('--input', type=str, required=True,
                        help='Path to image file or directory of images')
    parser.add_argument('--models-dir', type=str, default='output',
                        help='Directory containing trained models')
    parser.add_argument('--output-dir', type=str, default='results',
                        help='Directory to save visualizations')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to use (cuda:0, cpu, etc.)')
    
    args = parser.parse_args()
    main(args)
