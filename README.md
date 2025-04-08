# Plastic and Algae Detection in Oceans

This project implements a computer vision system for detecting plastic waste and algal blooms in ocean images. The system uses deep learning to identify and classify different types of marine pollution.

## Features

- Detection of plastic waste in ocean images
- Identification of algal blooms
- Real-time processing capabilities
- Support for both image and video input
- Visualization of detection results

## Installation

1. Clone this repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Place your ocean images in the `data/input` directory
2. Run the detection script:
```bash
python detect.py --input data/input --output data/output
```

## Project Structure

```
.
├── data/               # Input and output data
├── models/            # Trained model files
├── utils/             # Utility functions
├── detect.py          # Main detection script
├── requirements.txt   # Project dependencies
└── README.md          # Project documentation
```

## Requirements

- Python 3.8+
- OpenCV
- TensorFlow
- NumPy
- Matplotlib

## License

MIT License 
