# Cool vs Uncool Outfit Analyzer

A personalized AI-powered web application that learns your style preferences and analyzes outfit photos to determine if they match your definition of "cool" or "uncool".

## Features

- **Personalized AI Model**: Trained on your own labeled fashion images
- **Web Interface**: Clean, minimalist design for easy photo upload and analysis
- **Real-time Predictions**: Instant "cool" vs "uncool" analysis with confidence scores
- **Interactive Labeling**: Command-line tool for training your personal style model

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Label Your Images

Train the model by labeling images as cool or uncool:

```bash
python3 simple_labeler.py
```

### 3. Train Your Model

```bash
python3 train.py --epochs 20
```

### 4. Launch Web Application

```bash
python3 app.py
```

Open your browser to `http://localhost:3000` and start analyzing outfits!

## Project Structure

```
├── app.py                 # Flask web application
├── model.py              # CNN model definition and training
├── train.py              # Command-line training script
├── simple_labeler.py     # Interactive labeling tool
├── labeling_app.py       # GUI labeling interface
├── predictor.py          # Prediction interface
├── launcher.py           # Easy launcher script
├── requirements.txt      # Python dependencies
├── templates/
│   └── index.html       # Web interface
├── labels.json          # Your labeled data (created after labeling)
├── cool_uncool_model.pth # Trained model (created after training)
└── README.md            # This file
```

## How It Works

1. **Label Images**: Use the labeling tool to mark fashion images as cool or uncool based on your preferences
2. **Train Model**: The CNN learns patterns from your labeled data
3. **Web Analysis**: Upload outfit photos to get instant predictions
4. **Personalized Results**: Get "jovana thinks you're X% cool" feedback

## Usage

### Labeling Images
- Run `python3 simple_labeler.py`
- Images open in Preview automatically
- Type `c` for cool, `u` for uncool
- Navigate with arrow keys or `n`/`p`

### Training
- Requires at least 20-50 labeled images
- Run `python3 train.py` for command-line training
- Model automatically saves as `cool_uncool_model.pth`

### Web Interface
- Clean, minimalist design
- Drag & drop or click to upload photos
- Instant analysis with confidence scores
- Personal feedback: "jovana thinks you're X% cool"

## Technical Details

- **Model**: Custom CNN with 4 convolutional blocks
- **Framework**: PyTorch for deep learning
- **Web**: Flask for the web interface
- **Frontend**: HTML/CSS/JavaScript with Bootstrap
- **Data**: JSON format for labels, PyTorch for model weights

## Customization

### Model Architecture
Modify `model.py` to change the CNN architecture:
- Add more convolutional layers
- Adjust filter sizes
- Modify dropout rates

### Web Interface
Customize `templates/index.html`:
- Change styling and colors
- Modify the interface layout
- Add new features

### Training Parameters
Adjust `train.py` arguments:
- `--epochs`: Number of training epochs
- `--batch_size`: Batch size for training
- `--lr`: Learning rate

## Requirements

- Python 3.7+
- PyTorch 2.0+
- Flask 2.0+
- PIL/Pillow
- NumPy
- Matplotlib
- Scikit-learn

## License

This project is open source. Feel free to modify and distribute according to your needs.

## Contributing

Feel free to extend this system with:
- Different model architectures (ResNet, EfficientNet, etc.)
- Multi-class classification
- Web interface improvements
- Mobile app integration
- Cloud deployment

---

**Happy analyzing! 🎨✨**

Your personal style AI is ready to learn your preferences and help you discover cool fashion!