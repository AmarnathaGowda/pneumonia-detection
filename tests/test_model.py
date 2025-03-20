import unittest
import os
from data.dataset import ChestXRayDataset
from data.preprocessing import val_test_transforms

class TestData(unittest.TestCase):
    def setUp(self):
        self.data_dir = 'data/raw/chest_xray/test'
        self.dataset = ChestXRayDataset(self.data_dir, transform=val_test_transforms)

    def test_dataset_length(self):
        self.assertGreater(len(self.dataset), 0, "Dataset should not be empty")

    def test_image_loading(self):
        image, label = self.dataset[0]
        self.assertEqual(image.shape, (3, 224, 224), "Image should have shape [3, 224, 224]")
        self.assertIn(label, [0, 1], "Label should be 0 or 1")

if __name__ == '__main__':
    unittest.main()