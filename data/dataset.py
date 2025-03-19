import os 
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from data.preprocessing import train_transforms,val_test_transforms


class ChestXRayDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = ['NORMAL', 'PNEUMONIA']
        self.image_paths = []
        self.labels = []

        # Collect image paths and labels
        for label, class_name in enumerate(self.classes):
            class_dir = os.path.join(root_dir, class_name)
            for img_name in os.listdir(class_dir):
                if img_name.endswith(('.jpeg', '.jpg', '.png')):
                    self.image_paths.append(os.path.join(class_dir, img_name))
                    self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')  # Convert to RGB
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label  

def get_data_loaders(data_dir, batch_size=32):
    train_dataset = ChestXRayDataset(os.path.join(data_dir, 'train'), transform=train_transforms)
    val_dataset = ChestXRayDataset(os.path.join(data_dir, 'val'), transform=val_test_transforms)
    test_dataset = ChestXRayDataset(os.path.join(data_dir, 'test'), transform=val_test_transforms)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    data_dir = 'data/raw/chest_xray'
    train_loader, val_loader, test_loader = get_data_loaders(data_dir, batch_size=4)

    # Load one batch from the training loader
    images, labels = next(iter(train_loader))
    print(f"Batch shape: {images.shape}, Labels: {labels}")