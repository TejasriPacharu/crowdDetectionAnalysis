import os
import argparse
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

# Import models
from models.csrnet import csrnet
from models.density_classifier import density_classifier


def quantize_model(model, input_shape, output_path, backend='qnnpack'):
    """
    Quantize a PyTorch model for edge deployment
    
    Args:
        model: PyTorch model
        input_shape: Input shape (batch_size, channels, height, width)
        output_path: Path to save quantized model
        backend: Quantization backend ('qnnpack' for ARM, 'fbgemm' for x86)
    """
    # Set model to evaluation mode
    model.eval()
    
    # Set quantization backend
    torch.backends.quantized.engine = backend
    
    # Create example input
    example_input = torch.randn(input_shape)
    
    # Fuse modules where applicable (Conv2d + BatchNorm2d + ReLU)
    # This is model-specific and may need adjustment
    model_fused = torch.quantization.fuse_modules(model, [['conv', 'bn', 'relu']], inplace=False)
    
    # Prepare model for quantization
    model_prepared = torch.quantization.prepare(model_fused)
    
    # Calibrate with example data
    model_prepared(example_input)
    
    # Convert to quantized model
    model_quantized = torch.quantization.convert(model_prepared)
    
    # Save quantized model
    torch.jit.save(torch.jit.script(model_quantized), output_path)
    
    print(f"Quantized model saved to {output_path}")
    
    # Calculate size reduction
    original_size = os.path.getsize(output_path.replace('_quantized.pt', '.pt'))
    quantized_size = os.path.getsize(output_path)
    reduction = (1 - quantized_size / original_size) * 100
    
    print(f"Size reduction: {reduction:.2f}% ({original_size / 1e6:.2f} MB -> {quantized_size / 1e6:.2f} MB)")
    
    return model_quantized


def optimize_for_edge(model_dir, output_dir, target_device='arm'):
    """
    Optimize models for edge deployment
    
    Args:
        model_dir: Directory containing trained models
        output_dir: Directory to save optimized models
        target_device: Target device ('arm' or 'x86')
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Set quantization backend based on target device
    backend = 'qnnpack' if target_device == 'arm' else 'fbgemm'
    
    # Optimize CSRNet
    csrnet_path = os.path.join(model_dir, 'csrnet_best.pth')
    if os.path.exists(csrnet_path):
        print("Optimizing CSRNet...")
        
        # Load model
        model = csrnet(pretrained=False)
        model.load_state_dict(torch.load(csrnet_path, map_location='cpu'))
        
        # Export to ONNX
        onnx_path = os.path.join(output_dir, 'csrnet_optimized.onnx')
        dummy_input = torch.randn(1, 3, 384, 384)
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=12,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )
        print(f"CSRNet exported to ONNX: {onnx_path}")
        
        # Quantize model
        quantized_path = os.path.join(output_dir, 'csrnet_quantized.pt')
        quantize_model(model, (1, 3, 384, 384), quantized_path, backend=backend)
    
    # Optimize density classifier
    classifier_path = os.path.join(model_dir, 'classifier_best.pth')
    if os.path.exists(classifier_path):
        print("Optimizing density classifier...")
        
        # Load model
        model = density_classifier(pretrained=False)
        model.load_state_dict(torch.load(classifier_path, map_location='cpu'))
        
        # Export to ONNX
        onnx_path = os.path.join(output_dir, 'classifier_optimized.onnx')
        dummy_input = torch.randn(1, 3, 224, 224)
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=12,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )
        print(f"Density classifier exported to ONNX: {onnx_path}")
        
        # Quantize model
        quantized_path = os.path.join(output_dir, 'classifier_quantized.pt')
        quantize_model(model, (1, 3, 224, 224), quantized_path, backend=backend)
    
    print("Optimization completed!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Optimize models for edge deployment')
    
    parser.add_argument('--model-dir', type=str, required=True,
                        help='Directory containing trained models')
    parser.add_argument('--output-dir', type=str, default='optimized_models',
                        help='Directory to save optimized models')
    parser.add_argument('--target-device', type=str, default='arm', choices=['arm', 'x86'],
                        help='Target device (arm or x86)')
    
    args = parser.parse_args()
    optimize_for_edge(args.model_dir, args.output_dir, args.target_device)
