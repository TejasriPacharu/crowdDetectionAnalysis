import os
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Import models
from models.csrnet import csrnet
from models.density_classifier import density_classifier
from models.yolo_counter import YOLOCounter, prepare_yolo_dataset

# Import data loader
from data_loaders.ucf_qnrf_loader import get_dataloaders


def train_csrnet(model, train_loader, optimizer, criterion, device, epoch, writer):
    """
    Train CSRNet for one epoch
    """
    model.train()
    epoch_loss = 0
    
    with tqdm(train_loader, desc=f"CSRNet Epoch {epoch}") as pbar:
        for batch_idx, data in enumerate(pbar):
            # Get data
            images = data['image'].to(device)
            density_maps = data['density_map'].to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            
            # Calculate loss
            loss = criterion(outputs, density_maps)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Update statistics
            epoch_loss += loss.item()
            pbar.set_postfix(loss=loss.item())
            
            # Log to TensorBoard
            global_step = epoch * len(train_loader) + batch_idx
            writer.add_scalar('CSRNet/Train/Loss', loss.item(), global_step)
    
    avg_loss = epoch_loss / len(train_loader)
    print(f"CSRNet Epoch {epoch} - Average Loss: {avg_loss:.4f}")
    writer.add_scalar('CSRNet/Train/EpochLoss', avg_loss, epoch)
    
    return avg_loss


def train_density_classifier(model, train_loader, optimizer, criterion, device, epoch, writer):
    """
    Train density classifier for one epoch
    """
    model.train()
    epoch_loss = 0
    correct = 0
    total = 0
    
    with tqdm(train_loader, desc=f"Classifier Epoch {epoch}") as pbar:
        for batch_idx, data in enumerate(pbar):
            # Get data
            images = data['image'].to(device)
            labels = data['is_dense'].float().to(device).unsqueeze(1)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(images)
            
            # Calculate loss
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Update statistics
            epoch_loss += loss.item()
            
            # Calculate accuracy
            predicted = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            accuracy = 100 * correct / total
            pbar.set_postfix(loss=loss.item(), acc=f"{accuracy:.2f}%")
            
            # Log to TensorBoard
            global_step = epoch * len(train_loader) + batch_idx
            writer.add_scalar('Classifier/Train/Loss', loss.item(), global_step)
    
    avg_loss = epoch_loss / len(train_loader)
    accuracy = 100 * correct / total
    print(f"Classifier Epoch {epoch} - Average Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
    writer.add_scalar('Classifier/Train/EpochLoss', avg_loss, epoch)
    writer.add_scalar('Classifier/Train/Accuracy', accuracy, epoch)
    
    return avg_loss, accuracy


def validate(csrnet_model, classifier_model, yolo_counter, val_loader, device, epoch, writer):
    """
    Validate the hybrid model
    """
    csrnet_model.eval()
    classifier_model.eval()
    
    mae = 0
    mse = 0
    classifier_correct = 0
    classifier_total = 0
    
    with torch.no_grad():
        for batch_idx, data in enumerate(tqdm(val_loader, desc=f"Validation Epoch {epoch}")):
            # Get data
            images = data['image'].to(device)
            density_maps = data['density_map'].to(device)
            true_counts = data['count'].to(device)
            is_dense = data['is_dense'].float().to(device).unsqueeze(1)
            
            # Density classification
            density_pred = classifier_model(images)
            
            # Calculate classifier accuracy
            classifier_pred = (density_pred > 0.5).float()
            classifier_total += is_dense.size(0)
            classifier_correct += (classifier_pred == is_dense).sum().item()
            
            # Process each image in the batch
            pred_counts = []
            for i in range(images.size(0)):
                img = images[i].unsqueeze(0)
                
                # Decide which model to use based on density prediction
                if classifier_pred[i] > 0.5:  # Dense crowd
                    # Use CSRNet
                    density_map = csrnet_model(img)
                    count = density_map.sum().item()
                else:  # Sparse crowd
                    # Use YOLOv8
                    count, _ = yolo_counter.count(img.cpu().numpy(), device=device)
                    count = count[0]  # Get count from list
                
                pred_counts.append(count)
            
            # Convert to tensor
            pred_counts = torch.tensor(pred_counts, device=device)
            
            # Calculate MAE and MSE
            batch_mae = torch.abs(pred_counts - true_counts).mean().item()
            batch_mse = ((pred_counts - true_counts) ** 2).mean().item()
            
            mae += batch_mae
            mse += batch_mse
    
    # Calculate average metrics
    mae /= len(val_loader)
    mse /= len(val_loader)
    rmse = np.sqrt(mse)
    classifier_accuracy = 100 * classifier_correct / classifier_total
    
    print(f"Validation Epoch {epoch} - MAE: {mae:.2f}, RMSE: {rmse:.2f}, Classifier Accuracy: {classifier_accuracy:.2f}%")
    
    # Log to TensorBoard
    writer.add_scalar('Validation/MAE', mae, epoch)
    writer.add_scalar('Validation/RMSE', rmse, epoch)
    writer.add_scalar('Validation/ClassifierAccuracy', classifier_accuracy, epoch)
    
    return mae, rmse, classifier_accuracy


def main(args):
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create TensorBoard writer
    writer = SummaryWriter(os.path.join(args.output_dir, 'logs'))
    
    # Get data loaders
    train_loader, val_loader = get_dataloaders(
        root_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        density_map_size=(args.density_map_size, args.density_map_size)
    )
    
    print(f"Train dataset size: {len(train_loader.dataset)}")
    print(f"Validation dataset size: {len(val_loader.dataset)}")
    
    # Initialize models
    print("Initializing models...")
    
    # CSRNet for dense crowd counting
    csrnet_model = csrnet(pretrained=True).to(device)
    
    # Density classifier
    classifier_model = density_classifier(pretrained=True).to(device)
    
    # YOLOv8 for sparse crowd counting
    yolo_counter = YOLOCounter(model_size=args.yolo_model_size, pretrained=True)
    
    # Prepare YOLO dataset if needed
    if args.train_yolo:
        print("Preparing YOLO dataset...")
        yolo_data_dir = os.path.join(args.output_dir, 'yolo_dataset')
        data_yaml = prepare_yolo_dataset(train_loader.dataset, yolo_data_dir)
        print(f"YOLO dataset prepared at {yolo_data_dir}")
    
    # Define optimizers
    csrnet_optimizer = optim.Adam(csrnet_model.parameters(), lr=args.lr)
    classifier_optimizer = optim.Adam(classifier_model.parameters(), lr=args.lr)
    
    # Define learning rate schedulers
    csrnet_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        csrnet_optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    classifier_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        classifier_optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Define loss functions
    csrnet_criterion = nn.MSELoss()
    classifier_criterion = nn.BCELoss()
    
    # Train YOLOv8 if needed
    if args.train_yolo:
        print("Training YOLOv8...")
        yolo_counter.train(
            data_yaml=data_yaml,
            epochs=args.yolo_epochs,
            batch_size=args.batch_size,
            imgsz=args.yolo_img_size,
            device=args.device
        )
        print("YOLOv8 training completed")
    
    # Training loop
    print("Starting training...")
    best_mae = float('inf')
    
    for epoch in range(1, args.epochs + 1):
        # Train CSRNet
        csrnet_loss = train_csrnet(
            model=csrnet_model,
            train_loader=train_loader,
            optimizer=csrnet_optimizer,
            criterion=csrnet_criterion,
            device=device,
            epoch=epoch,
            writer=writer
        )
        
        # Train density classifier
        classifier_loss, classifier_acc = train_density_classifier(
            model=classifier_model,
            train_loader=train_loader,
            optimizer=classifier_optimizer,
            criterion=classifier_criterion,
            device=device,
            epoch=epoch,
            writer=writer
        )
        
        # Validate
        mae, rmse, classifier_val_acc = validate(
            csrnet_model=csrnet_model,
            classifier_model=classifier_model,
            yolo_counter=yolo_counter,
            val_loader=val_loader,
            device=device,
            epoch=epoch,
            writer=writer
        )
        
        # Update learning rate schedulers
        csrnet_scheduler.step(csrnet_loss)
        classifier_scheduler.step(classifier_loss)
        
        # Save best model
        if mae < best_mae:
            best_mae = mae
            print(f"New best MAE: {best_mae:.2f}, saving models...")
            
            # Save CSRNet
            torch.save(csrnet_model.state_dict(), os.path.join(args.output_dir, 'csrnet_best.pth'))
            
            # Save density classifier
            torch.save(classifier_model.state_dict(), os.path.join(args.output_dir, 'classifier_best.pth'))
        
        # Save checkpoint every few epochs
        if epoch % args.save_interval == 0:
            print(f"Saving checkpoint at epoch {epoch}...")
            
            # Save CSRNet
            torch.save({
                'epoch': epoch,
                'model_state_dict': csrnet_model.state_dict(),
                'optimizer_state_dict': csrnet_optimizer.state_dict(),
                'loss': csrnet_loss,
            }, os.path.join(args.output_dir, f'csrnet_epoch_{epoch}.pth'))
            
            # Save density classifier
            torch.save({
                'epoch': epoch,
                'model_state_dict': classifier_model.state_dict(),
                'optimizer_state_dict': classifier_optimizer.state_dict(),
                'loss': classifier_loss,
                'accuracy': classifier_acc,
            }, os.path.join(args.output_dir, f'classifier_epoch_{epoch}.pth'))
    
    # Save final models
    print("Training completed, saving final models...")
    
    # Save CSRNet
    torch.save(csrnet_model.state_dict(), os.path.join(args.output_dir, 'csrnet_final.pth'))
    
    # Save density classifier
    torch.save(classifier_model.state_dict(), os.path.join(args.output_dir, 'classifier_final.pth'))
    
    # Export models for deployment
    if args.export_models:
        print("Exporting models for deployment...")
        
        # Export YOLOv8
        yolo_path = yolo_counter.export(format='onnx')
        print(f"YOLOv8 exported to {yolo_path}")
        
        # Export CSRNet
        csrnet_model.eval()
        dummy_input = torch.randn(1, 3, args.density_map_size, args.density_map_size).to(device)
        csrnet_path = os.path.join(args.output_dir, 'csrnet.onnx')
        torch.onnx.export(
            csrnet_model,
            dummy_input,
            csrnet_path,
            export_params=True,
            opset_version=12,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )
        print(f"CSRNet exported to {csrnet_path}")
        
        # Export density classifier
        classifier_model.eval()
        dummy_input = torch.randn(1, 3, 224, 224).to(device)
        classifier_path = os.path.join(args.output_dir, 'classifier.onnx')
        torch.onnx.export(
            classifier_model,
            dummy_input,
            classifier_path,
            export_params=True,
            opset_version=12,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )
        print(f"Density classifier exported to {classifier_path}")
    
    print("All done!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train hybrid crowd counting model')
    
    # Dataset parameters
    parser.add_argument('--data-dir', type=str, default='UCF-QNRF_ECCV18',
                        help='Path to UCF-QNRF dataset')
    parser.add_argument('--density-map-size', type=int, default=384,
                        help='Size of density map')
    
    # Training parameters
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-5,
                        help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda:0',
                        help='Device to use (cuda:0, cuda:1, cpu, etc.)')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of workers for data loading')
    parser.add_argument('--save-interval', type=int, default=10,
                        help='Save checkpoint every N epochs')
    
    # YOLOv8 parameters
    parser.add_argument('--train-yolo', action='store_true',
                        help='Whether to train YOLOv8')
    parser.add_argument('--yolo-model-size', type=str, default='n',
                        help='YOLOv8 model size (n, s, m, l, x)')
    parser.add_argument('--yolo-epochs', type=int, default=50,
                        help='Number of epochs for YOLOv8 training')
    parser.add_argument('--yolo-img-size', type=int, default=640,
                        help='Image size for YOLOv8 training')
    
    # Output parameters
    parser.add_argument('--output-dir', type=str, default='output',
                        help='Output directory')
    parser.add_argument('--export-models', action='store_true',
                        help='Export models for deployment')
    
    args = parser.parse_args()
    main(args)
