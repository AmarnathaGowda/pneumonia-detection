# Pneumonia Detection with PyTorch

This project builds an end-to-end machine learning system to classify chest X-ray images as "Normal" or "Pneumonia" using PyTorch, Flask, and a web frontend. It includes data preprocessing, model training, evaluation, interpretability (Grad-CAM), hyperparameter tuning, API deployment, and testing.

## Prerequisites
- Python 3.8+
- Virtual environment (recommended)
- GPU (optional, for faster training)

## Installation

### 1. Clone the repository:
```bash
git clone <repository-url>
cd pneumonia-detection
```

### 2. Set up a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # macOS/Linux
venv\Scripts\activate     # Windows
```

### 3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Dataset
**Source**: Chest X-Ray Images (Pneumonia) dataset from Kaggle.

**Setup**: Run the data acquisition script:
```bash
bash scripts/download_data.sh
```
> Requires Kaggle API credentials (`kaggle.json` in `~/.kaggle/`).

## Usage

### Train the Model:
```bash
python model/train.py
```
> Trains the model and saves it as `model/pneumonia_classifier.pth`.

### Tune Hyperparameters:
```bash
python model/tune.py
```
> Optimizes hyperparameters and saves the best model as `model/optimized_pneumonia_classifier.pth`.

### Run the API:
```bash
python api/app.py
```
> Starts the prediction API at `http://localhost:5000`.

### Run the Frontend:
```bash
python frontend/app.py
```
> Starts the web interface at `http://localhost:5001`.

### Test the System:
```bash
python -m unittest discover -s tests
```

## Project Structure
```
pneumonia-detection/
├── data/                # Data loading and preprocessing
├── model/               # Model architecture, training, and evaluation
├── api/                 # Flask API for predictions
├── frontend/            # Web interface
├── utils/               # Helper functions (config, logging)
├── scripts/             # Automation scripts
├── tests/               # Unit tests
├── requirements.txt     # Dependencies
├── README.md            # Project documentation
```

## API Endpoint
**POST /predict**: Upload an image to get a prediction.

Example with `curl`:
```bash
curl -X POST -F "image=@path/to/image.jpeg" http://localhost:5000/predict
```

## Frontend
**Access at**: `http://localhost:5001`

Upload a chest X-ray image to see the prediction and confidence score.

## Contributing
- Report issues or suggest improvements via pull requests.
- Ensure tests pass before submitting changes.

## License
MIT License (or specify your preferred license).

## Clean Up Unnecessary Files
### Remove temporary or redundant files:
```bash
find . -name "*.pyc" -delete          # macOS/Linux
find . -name "__pycache__" -delete    # macOS/Linux
del /s *.pyc                          # Windows (Command Prompt)
rmdir /s /q __pycache__               # Windows (Command Prompt)
```
> Remove `api.log` if not needed:
```bash
rm api.log  # macOS/Linux
del api.log # Windows
```

### Review Saved Models:
Keep only `optimized_pneumonia_classifier.pth` unless `pneumonia_classifier.pth` is needed:
```bash
rm model/pneumonia_classifier.pth  # macOS/Linux
del model\pneumonia_classifier.pth # Windows
```

### Verify Scripts:
Ensure `scripts/download_data.sh` is functional:
```bash
echo "kaggle datasets download -d paultimothymooney/chest-xray-pneumonia --path data/raw" > scripts/download_data.sh
chmod +x scripts/download_data.sh  # macOS/Linux
```

## Verify Project Stability
### Test Data and Model:
```bash
python -m unittest discover -s tests
```

### Run API and Frontend:
In one terminal:
```bash
python api/app.py
```
In another:
```bash
python frontend/app.py
```
Open `http://localhost:5001`, upload an image, and confirm the prediction works.

## Final Notes
### Version Control:
If using Git, commit your changes:
```bash
git add .
git commit -m "Completed pneumonia detection project with documentation"
```

### Deployment:
The `scripts/deploy.sh` file is a placeholder. For real deployment (e.g., to AWS), you’d need to add cloud-specific commands.

### Future Enhancements:
Consider adding:
- Model versioning
- User authentication
- A more advanced frontend

