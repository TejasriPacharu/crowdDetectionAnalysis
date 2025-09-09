import os
import random
import numpy as np
import scipy.io as sio
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import cv2
from scipy.ndimage import gaussian_filter


def create_density_map(points, shape, sigma=15):
    """
    Generate density map from point annotations
    
    Args:
        points: Nx2 array of point coordinates (x, y)
        shape: (height, width) of the output density map
        sigma: Gaussian kernel standard deviation
        
    Returns:
        Density map where sum equals the count of points
    """
    density_map = np.zeros(shape, dtype=np.float32)
    
    # If no points, return empty density map
    if points.shape[0] == 0:
        return density_map
    
    # Create density map by placing Gaussian kernels at each point
    for i in range(points.shape[0]):
        x, y = int(points[i, 0]), int(points[i, 1])
        
        # Skip points outside the image
        if x < 0 or x >= shape[1] or y < 0 or y >= shape[0]:
            continue
            
        # Place a single 1 at the point location
        density_map[y, x] = 1
    
    # Apply Gaussian filter to spread the point
    density_map = gaussian_filter(density_map, sigma=sigma)
    
    # Normalize the density map so its sum equals the count of points
    if density_map.sum() > 0:
        density_map = density_map * (points.shape[0] / density_map.sum())
    
    return density_map


class UCFQNRFDataset(Dataset):
    """
    Dataset class for UCF-QNRF dataset
    """
    def __init__(self, root_dir, phase='train', transform=None, target_transform=None, 
                 density_map_sigma=15, density_map_size=(384, 384), 
                 classification_threshold=100):
        """
        Args:
            root_dir: Root directory of UCF-QNRF dataset
            phase: 'train' or 'test'
            transform: Transform to apply to images
            target_transform: Transform to apply to targets
            density_map_sigma: Sigma for Gaussian kernel in density map
            density_map_size: Size of density map (height, width)
            classification_threshold: Threshold for classifying dense vs sparse crowds
        """
        self.root_dir = root_dir
        self.phase = phase.capitalize()  # 'Train' or 'Test'
        self.transform = transform
        self.target_transform = target_transform
        self.density_map_sigma = density_map_sigma
        self.density_map_size = density_map_size
        self.classification_threshold = classification_threshold
        
        # Get list of all images
        self.image_files = []
        phase_dir = os.path.join(root_dir, self.phase)
        
        for filename in os.listdir(phase_dir):
            if filename.endswith('.jpg'):
                self.image_files.append(filename)
        
        self.image_files.sort()
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Get image path
        img_name = self.image_files[idx]
        img_path = os.path.join(self.root_dir, self.phase, img_name)
        
        # Get annotation path
        ann_name = img_name.replace('.jpg', '_ann.mat')
        ann_path = os.path.join(self.root_dir, self.phase, ann_name)
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        width, height = image.size
        
        # Load annotation
        ann_data = sio.loadmat(ann_path)
        points = ann_data['annPoints']  # Nx2 array with (x, y) coordinates
        
        # Count of people in the image
        count = points.shape[0]
        
        # Create density map
        density_map = create_density_map(points, (height, width), sigma=self.density_map_sigma)
        
        # Create binary classification label (0: sparse, 1: dense)
        is_dense = 1 if count >= self.classification_threshold else 0
        
        # Create detection targets for YOLOv8 (for sparse crowds)
        # For simplicity, we'll use a fixed-size bounding box around each point
        # In a real implementation, you might want to use actual person bounding boxes
        yolo_targets = []
        box_size = 50  # Fixed box size in pixels
        
        for i in range(points.shape[0]):
            x, y = points[i, 0], points[i, 1]
            
            # Create a bounding box around the point
            x1 = max(0, x - box_size // 2)
            y1 = max(0, y - box_size // 2)
            x2 = min(width - 1, x + box_size // 2)
            y2 = min(height - 1, y + box_size // 2)
            
            # Normalize coordinates to [0, 1]
            x1, y1, x2, y2 = x1/width, y1/height, x2/width, y2/height
            
            # YOLO format: class_id, x_center, y_center, width, height
            class_id = 0  # 0 for person
            x_center = (x1 + x2) / 2
            y_center = (y1 + y2) / 2
            box_width = x2 - x1
            box_height = y2 - y1
            
            yolo_targets.append([class_id, x_center, y_center, box_width, box_height])
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Convert density map to tensor
        density_map = torch.from_numpy(density_map).float().unsqueeze(0)  # Add channel dimension
        
        # Resize density map if needed
        if self.density_map_size != (height, width):
            density_map = torch.nn.functional.interpolate(
                density_map.unsqueeze(0), 
                size=self.density_map_size, 
                mode='bilinear', 
                align_corners=False
            ).squeeze(0)
        
        # Convert YOLO targets to tensor
        yolo_targets = torch.tensor(yolo_targets).float() if yolo_targets else torch.zeros((0, 5)).float()
        
        return {
            'image': image,
            'density_map': density_map,
            'count': count,
            'is_dense': is_dense,
            'yolo_targets': yolo_targets,
            'image_path': img_path
        }


def custom_collate_fn(batch):
    """
    Custom collate function to handle variable-sized tensors
    
    Args:
        batch: List of samples from dataset
        
    Returns:
        Batched tensors with padding for variable-sized data
    """
    # Convert batch from list of dicts to dict of lists
    keys = batch[0].keys()
    collated_batch = {key: [] for key in keys}
    
    for sample in batch:
        for key in keys:
            collated_batch[key].append(sample[key])
    
    # Stack fixed-size tensors
    collated_batch['image'] = torch.stack(collated_batch['image'], 0)
    collated_batch['density_map'] = torch.stack(collated_batch['density_map'], 0)
    collated_batch['count'] = torch.tensor(collated_batch['count'])
    collated_batch['is_dense'] = torch.tensor(collated_batch['is_dense'])
    
    # Keep variable-sized data as lists
    # We don't stack 'yolo_targets' as they have variable sizes
    
    return collated_batch


def get_dataloaders(root_dir, batch_size=8, num_workers=4, density_map_size=(384, 384)):
    """
    Create data loaders for training and testing
    
    Args:
        root_dir: Root directory of UCF-QNRF dataset
        batch_size: Batch size
        num_workers: Number of workers for data loading
        density_map_size: Size of density map (height, width)
        
    Returns:
        train_loader, test_loader
    """
    # Define transforms
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(size=384, scale=(0.75, 1.0)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = UCFQNRFDataset(
        root_dir=root_dir,
        phase='train',
        transform=train_transform,
        density_map_size=density_map_size
    )
    
    test_dataset = UCFQNRFDataset(
        root_dir=root_dir,
        phase='test',
        transform=test_transform,
        density_map_size=density_map_size
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        collate_fn=custom_collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=custom_collate_fn
    )
    
    return train_loader, test_loader
