import unittest
from api.app import app
from flask import json
import os

class TestAPI(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True
        self.sample_image = 'data/raw/chest_xray/test/NORMAL/IM-0001-0001.jpeg'

    def test_predict_endpoint_success(self):
        with open(self.sample_image, 'rb') as img:
            response = self.app.post('/predict', data={'image': (img, 'test.jpeg')},
                                    content_type='multipart/form-data')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('prediction', data)
        self.assertIn('confidence', data)
        self.assertIn(data['prediction'], ['Normal', 'Pneumonia'])

    def test_predict_endpoint_no_image(self):
        response = self.app.post('/predict', content_type='multipart/form-data')
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertIn('error', data)

if __name__ == '__main__':
    unittest.main()