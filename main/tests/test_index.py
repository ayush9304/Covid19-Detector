from django.test import TestCase, Client
import os

class BaseTest(TestCase):
    def setUp(self):
        self.client = Client()
        self.test_image_path = 'main/tests/imgs/test_image2.png'
        self.test_image_data = open(self.test_image_path, 'rb')

        self.test_image2_path = 'main/tests/imgs/test_image_nx.jpg'
        self.test_image2_data = open(self.test_image2_path, 'rb')
        
        return super().setUp()

    def test_index(self):
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'main/index.html')
        self.assertContains(response, '<title>COVID-19 Detector | CoviScan</title>')
        self.assertContains(response, '<h5 class="card-header bg-light">COVID-19 Detection Using X-Rays</h5>')
        self.assertTemplateUsed(response,'main/index.html')

    def test_index_predict_get(self):
        response = self.client.get('/predict')
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, '"success": false')
        self.assertContains(response, '"description": "Request method must be POST"')

        response = self.client.get('/predict', {'image': self.test_image_data, 'name': 'Test'})
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, '"success": false')
        self.assertContains(response, '"description": "Request method must be POST"')
    
    def test_index_predict_post(self):
        response = self.client.post('/predict', {'image': self.test_image_data, 'name': 'Test'})
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, '"success": true')
        self.assertContains(response, '"description": "Successfully uploaded"')

        response = self.client.post('/predict', {'image': self.test_image2_data, 'name': 'TestNX'})
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, '"success": false')
