# Hybrid Crowd Counting System

This project implements a hybrid deep learning system for crowd counting that combines:
- CSRNet for dense crowd density map estimation
- YOLOv8 for sparse crowd detection
- A density-classifier CNN to automatically switch between the two models

The system is designed to work with the UCF-QNRF dataset and is optimized for IoT edge deployment with cloud analytics support.

## Project Structure

```
crowd_counting/
├── data_loaders/
│   └── ucf_qnrf_loader.py    # Data loader for UCF-QNRF dataset
├── models/
│   ├── csrnet.py             # CSRNet model for dense crowd counting
│   ├── density_classifier.py # Density classifier model
│   └── yolo_counter.py       # YOLOv8 integration for sparse crowd counting
├── utils/
│   └── optimize_for_edge.py  # Utilities for optimizing models for edge deployment
└── train.py                  # Main training script
```

## Environment Setup

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (recommended)
- UCF-QNRF dataset

### Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/crowd-counting.git
cd crowd-counting
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

4. Download the UCF-QNRF dataset and place it in the project directory.

## Training the Model

### Basic Training

To train the hybrid crowd counting model with default parameters:

```bash
python crowd_counting/train.py --data-dir UCF-QNRF_ECCV18 --output-dir output
```

### Advanced Training Options

```bash
python crowd_counting/train.py \
    --data-dir UCF-QNRF_ECCV18 \
    --output-dir output \
    --batch-size 8 \
    --epochs 100 \
    --lr 1e-5 \
    --device cuda:0 \
    --train-yolo \
    --yolo-model-size n \
    --yolo-epochs 50 \
    --export-models
```

### Training Parameters

- `--data-dir`: Path to UCF-QNRF dataset
- `--output-dir`: Directory to save output models and logs
- `--batch-size`: Batch size for training
- `--epochs`: Number of training epochs
- `--lr`: Learning rate
- `--device`: Device to use for training (e.g., cuda:0, cuda:1, cpu)
- `--train-yolo`: Whether to train YOLOv8 (otherwise uses pretrained weights)
- `--yolo-model-size`: YOLOv8 model size (n, s, m, l, x)
- `--yolo-epochs`: Number of epochs for YOLOv8 training
- `--export-models`: Export models for deployment after training

## Optimizing for Edge Deployment

After training, you can optimize the models for edge deployment:

```bash
python crowd_counting/utils/optimize_for_edge.py \
    --model-dir output \
    --output-dir optimized_models \
    --target-device arm
```

### Optimization Parameters

- `--model-dir`: Directory containing trained models
- `--output-dir`: Directory to save optimized models
- `--target-device`: Target device (arm or x86)

## Monitoring Training

You can monitor the training progress using TensorBoard:

```bash
tensorboard --logdir output/logs
```

Then open your browser and navigate to http://localhost:6006.

## Evaluation

The training script automatically evaluates the model on the validation set after each epoch. The best model is saved based on the Mean Absolute Error (MAE) metric.

## IoT Edge Deployment

The optimized models can be deployed to IoT edge devices using frameworks like TensorRT, ONNX Runtime, or TFLite. The `optimize_for_edge.py` script exports the models to ONNX format and creates quantized versions for efficient deployment.

## Cloud Analytics

The models can be integrated with cloud services for analytics. The exported ONNX models are compatible with most cloud AI platforms.

## Citation

If you use this code for your research, please cite the following papers:

```
@inproceedings{idrees2018composition,
  title={Composition Loss for Counting, Density Map Estimation and Localization in Dense Crowds},
  author={Idrees, Haroon and Tayyab, Muhmmad and Athrey, Kishan and Zhang, Dong and Al-Maadeed, Somaya and Rajpoot, Nasir and Shah, Mubarak},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
  year={2018}
}

@inproceedings{li2018csrnet,
  title={CSRNet: Dilated convolutional neural networks for understanding the highly congested scenes},
  author={Li, Yuhong and Zhang, Xiaofan and Chen, Deming},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year={2018}
}
```
