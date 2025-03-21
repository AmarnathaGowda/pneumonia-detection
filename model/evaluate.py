import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from torchvision import transforms
from PIL import Image

import sys,os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.architecture import PneumoniaClassifier
from utils.config import DEVICE,NUM_CLASSES,BATCH_SIZE,DATA_PATH
from data.dataset import get_data_loaders

from utils.gradcam import generate_gradcam
from data.preprocessing import val_test_transforms



# Load the model
model = PneumoniaClassifier(num_classes=NUM_CLASSES).to(DEVICE)
model.load_state_dict(torch.load('model/pneumonia_classifier.pth'))
model.eval()  # Set to evaluation mode

# Get the test loader
_, _, test_loader = get_data_loaders(DATA_PATH, batch_size=BATCH_SIZE)

# Evaluate the Model
all_preds = []
all_labels = []

with torch.no_grad():  # Disable gradient computation
    for images, labels in test_loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        probs = F.softmax(outputs, dim=1)  # Convert logits to probabilities
        _, preds = torch.max(probs, 1)  # Get the predicted class
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Calculate metrics
accuracy = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds)
recall = recall_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds)

print(f"Test Accuracy: {accuracy:.4f}")
print(f"Test Precision: {precision:.4f}")
print(f"Test Recall: {recall:.4f}")
print(f"Test F1-Score: {f1:.4f}")

# Visualize the Results
# Compute confusion matrix
cm = confusion_matrix(all_labels, all_preds)

# Plot confusion matrix
plt.figure(figsize=(6,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Normal', 'Pneumonia'], yticklabels=['Normal', 'Pneumonia'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
# Save confusion matrix plot
plt.savefig('confusion_matrix.png')

# Save the Results
# Save metrics to a text file
with open('evaluation_results.txt', 'w') as f:
    f.write(f"Test Accuracy: {accuracy:.4f}\n")
    f.write(f"Test Precision: {precision:.4f}\n")
    f.write(f"Test Recall: {recall:.4f}\n")
    f.write(f"Test F1-Score: {f1:.4f}\n")



# Load a sample image
# sample_image_path = 'data/raw/chest_xray/test/PNEUMONIA/person1_bacteria_1.jpeg'
sample_image_path = '/Users/amarnathgowda/Desktop/pojects/pneumonia-detection/data/raw/chest_xray/test/PNEUMONIA/person1_virus_6.jpeg'
image = Image.open(sample_image_path).convert('RGB')
image_tensor = val_test_transforms(image)

# Generate Grad-CAM for class 1 (Pneumonia)
cam_image = generate_gradcam(model, image_tensor, target_class=1, device=DEVICE)

# Display the result
plt.imshow(cam_image)
plt.title('Grad-CAM Heatmap')
plt.show()
plt.savefig('gradcam_sample.png')
