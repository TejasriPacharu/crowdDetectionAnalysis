import os
import torch
import numpy as np
from ultralytics import YOLO


class YOLOCounter:
    """
    YOLOv8 integration for sparse crowd counting
    
    This class wraps the YOLOv8 model for person detection and counting
    """
    def __init__(self, model_size='n', pretrained=True, confidence=0.25):
        """
        Initialize YOLOv8 counter
        
        Args:
            model_size: YOLOv8 model size ('n', 's', 'm', 'l', 'x')
            pretrained: Whether to use pretrained weights
            confidence: Confidence threshold for detections
        """
        self.confidence = confidence
        
        # Load YOLOv8 model
        if pretrained:
            self.model = YOLO(f'yolov8{model_size}.pt')
        else:
            self.model = YOLO(f'yolov8{model_size}.yaml')
        
        # Set model parameters
        self.model.conf = confidence  # Confidence threshold
        self.model.iou = 0.45  # IoU threshold
        self.model.classes = [0]  # Only detect people (class 0 in COCO)
    
    def count(self, images, device=None):
        """
        Count people in images using YOLOv8
        
        Args:
            images: List of images or batch tensor
            device: Device to run inference on
            
        Returns:
            counts: List of people counts for each image
            boxes: List of detection boxes for each image
        """
        # Run inference
        results = self.model(images, device=device)
        
        counts = []
        boxes_list = []
        
        # Process results
        for result in results:
            # Get boxes for class 0 (person)
            boxes = result.boxes
            person_boxes = boxes[boxes.cls == 0]
            
            # Count people
            count = len(person_boxes)
            counts.append(count)
            
            # Store boxes
            boxes_list.append(person_boxes)
        
        return counts, boxes_list
    
    def train(self, data_yaml, epochs=100, batch_size=16, imgsz=640, device=None):
        """
        Train YOLOv8 model
        
        Args:
            data_yaml: Path to data YAML file
            epochs: Number of training epochs
            batch_size: Batch size
            imgsz: Image size
            device: Device to train on
            
        Returns:
            Training metrics
        """
        # Train the model
        results = self.model.train(
            data=data_yaml,
            epochs=epochs,
            batch=batch_size,
            imgsz=imgsz,
            device=device,
            patience=20,  # Early stopping patience
            save=True,  # Save best model
            project='crowd_counting/runs',
            name='yolov8_train'
        )
        
        return results
    
    def export(self, format='onnx', dynamic=True):
        """
        Export YOLOv8 model to different formats for deployment
        
        Args:
            format: Export format ('onnx', 'tflite', 'coreml', etc.)
            dynamic: Whether to use dynamic axes
            
        Returns:
            Path to exported model
        """
        # Export the model
        path = self.model.export(format=format, dynamic=dynamic)
        return path


def prepare_yolo_dataset(dataset, output_dir):
    """
    Prepare YOLOv8 dataset from UCF-QNRF dataset
    
    Args:
        dataset: UCFQNRFDataset instance
        output_dir: Output directory for YOLO dataset
        
    Returns:
        Path to data YAML file
    """
    # Create directories
    os.makedirs(os.path.join(output_dir, 'images', 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'images', 'val'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'labels', 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'labels', 'val'), exist_ok=True)
    
    # Process dataset
    for i in range(len(dataset)):
        data = dataset[i]
        
        # Skip dense crowds
        if data['is_dense'] == 1:
            continue
        
        # Get image path and YOLO targets
        img_path = data['image_path']
        yolo_targets = data['yolo_targets'].numpy()
        
        # Determine if this is a training or validation sample (80/20 split)
        is_train = np.random.rand() < 0.8
        split = 'train' if is_train else 'val'
        
        # Copy image to YOLO dataset
        img_filename = os.path.basename(img_path)
        dst_img_path = os.path.join(output_dir, 'images', split, img_filename)
        
        # Create symbolic link to original image
        if not os.path.exists(dst_img_path):
            os.symlink(img_path, dst_img_path)
        
        # Create YOLO label file
        label_filename = img_filename.replace('.jpg', '.txt')
        label_path = os.path.join(output_dir, 'labels', split, label_filename)
        
        # Write YOLO labels
        with open(label_path, 'w') as f:
            for target in yolo_targets:
                # YOLO format: class_id, x_center, y_center, width, height
                line = ' '.join(map(str, target.tolist()))
                f.write(line + '\n')
    
    # Create data YAML file
    yaml_path = os.path.join(output_dir, 'data.yaml')
    with open(yaml_path, 'w') as f:
        f.write(f"path: {output_dir}\n")
        f.write("train: images/train\n")
        f.write("val: images/val\n")
        f.write("test: images/val\n")
        f.write("\n")
        f.write("nc: 1\n")  # Number of classes
        f.write("names: ['person']\n")  # Class names
    
    return yaml_path
