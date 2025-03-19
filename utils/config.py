import torch

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model hyperparameters
NUM_CLASSES = 2
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 10