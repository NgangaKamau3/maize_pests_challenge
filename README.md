# Fall Armyworm Detection System

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.6+](https://img.shields.io/badge/python-3.6+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?&logo=PyTorch&logoColor=white)](https://pytorch.org/)

A deep learning system for detecting fall armyworm infestation in maize plants using image classification, optimized for mobile deployment.

## 🌟 Features

- Mobile-optimized MobileNetV2 architecture
- High AUC score for accurate detection
- ONNX and quantization support for edge deployment
- Comprehensive data preprocessing pipeline
- Detailed evaluation metrics and visualizations

## 📊 Dataset Structure

The dataset is organized as follows:

```
PestDataset/
└── Combined_pestDataset/
    ├── Images/           # Contains all image files
    ├── Train.csv         # Training data with Image_id and Label columns
    ├── Test.csv          # Test data with Image_id column
    └── SampleSubmission.csv  # Sample submission format
```

## 📥 Dataset Setup

The dataset is not included in this repository due to size constraints. To set up the dataset:

1. Download the Fall Armyworm dataset from [source/link]
2. Extract the contents to `PestDataset/Combined_pestDataset/` in the project root
3. Verify the structure matches the expected format described above

## 🚀 Quick Start

### Prerequisites

```bash
pip install -r requirements.txt
```

### Training

```bash
python train.py
```

### Generate Submission

```bash
python evaluation.py
```

### Optimize for Mobile

```bash
python quantize.py
```

## 📋 Project Structure

```
├── preprocessing.py      # Data loading and augmentation
├── training.py           # Model definition and training
├── train.py              # Training script with CLI options
├── evaluation.py         # Evaluation and submission generation
├── generate_submission.py # Generate submission file
├── quantize.py           # Model optimization for mobile
├── main.py               # Main inference script
├── requirements.txt      # Dependencies
├── models/               # Saved model checkpoints
├── results/              # Evaluation results
├── LICENSE               # MIT License
└── CONTRIBUTING.md       # Contribution guidelines
```

## 📈 Performance

The model achieves a high AUC score on the test set, making it suitable for real-world deployment. The MobileNetV2 architecture ensures efficient inference on resource-constrained devices.

## 🤝 Contributing

Contributions are welcome! Please check out our [contribution guidelines](CONTRIBUTING.md).

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Contact

For questions or feedback, please open an issue on GitHub.

## 🔄 Development Workflow

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 🧪 Testing

Run tests with:

```bash
# Future implementation
# pytest
```

## 📱 Mobile Deployment

The model is optimized for mobile deployment using ONNX and quantization:

1. Train the model with `python train.py`
2. Optimize for mobile with `python quantize.py`
3. Use the generated `.onnx` file in your mobile application