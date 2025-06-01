# Fall Armyworm Detection System: User Guide

## Introduction

The Fall Armyworm Detection System is a mobile-optimized tool that helps farmers and agricultural workers identify fall armyworm infestations in maize plants using image classification. This guide provides instructions for installing, using, and troubleshooting the system.

## Installation

### Prerequisites

- Python 3.6 or higher
- Camera-equipped smartphone or tablet (for mobile deployment)
- Internet connection (for initial setup)

### Desktop Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/fall_armyworm_detection.git
   cd fall_armyworm_detection
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Verify installation:
   ```bash
   python main.py --help
   ```

### Mobile Setup

1. Download the pre-built mobile application from the releases page
2. Install the application on your Android or iOS device
3. Grant camera and storage permissions when prompted

## Quick Start

### Training a Model

If you want to train your own model:

```bash
python train.py --epochs 50 --batch_size 32
```

This will:
- Load the dataset from `PestDataset/Combined_pestDataset/`
- Train a MobileNetV2 model for 50 epochs
- Save the best model to `models/best_model.pth`

### Running Inference

To classify new images:

```bash
python main.py --model_path ./models/best_model.pth --test_csv ./path/to/images.csv
```

### Mobile Application

1. Open the Fall Armyworm Detection app
2. Choose either:
   - **Camera Mode**: Take a photo of a maize leaf
   - **Gallery Mode**: Select an existing image
3. Tap "Analyze" to process the image
4. View results showing:
   - Classification (Healthy/Infected)
   - Confidence percentage
   - Recommended actions

## Features

### Image Classification

- Binary classification of maize leaves (healthy vs. infected)
- High accuracy (>90%) in field conditions
- Fast inference (<1 second on most devices)

### Offline Functionality

- Once installed, the app works without internet connection
- All processing happens on-device for privacy and reliability

### Results Management

- Save and organize scan results
- Export reports as CSV or PDF
- Share findings via email or messaging apps

## Troubleshooting

### Common Issues

1. **Poor Detection Accuracy**
   - Ensure good lighting conditions
   - Hold camera 15-20cm from the leaf
   - Avoid shadows and glare
   - Clean camera lens

2. **App Crashes**
   - Ensure your device meets minimum requirements
   - Update to the latest app version
   - Clear app cache in device settings

3. **Slow Performance**
   - Close background applications
   - Restart the app
   - Ensure adequate storage space

### Getting Help

- Report issues on GitHub: https://github.com/yourusername/fall_armyworm_detection/issues
- Email support: support@example.com
- Community forum: https://community.example.com/fall-armyworm-detection

## Best Practices

1. **Image Capture**
   - Focus on affected areas of the leaf
   - Include some healthy tissue for comparison
   - Use natural daylight when possible
   - Hold the device steady

2. **Regular Monitoring**
   - Scan fields weekly during growing season
   - Increase frequency during high-risk periods
   - Monitor edges of fields more frequently

3. **Integrated Management**
   - Use detection results to inform treatment decisions
   - Combine with other monitoring methods
   - Follow local agricultural extension recommendations

## Privacy and Data

- All image processing occurs on your device
- No images are uploaded to servers without permission
- Optional: Share anonymized data to improve the system

## Updates and Maintenance

- Check for app updates monthly
- Model updates are released quarterly
- Enable automatic updates for best performance