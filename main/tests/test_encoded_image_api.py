from django.test import TestCase, Client
import base64
import os

class BaseTestCase(TestCase):
    def setUp(self):
        self.client = Client()
        self.test_image_path = 'main/tests/imgs/test_image2.png'
        self.test_image_data = open(self.test_image_path, 'rb').read()
        self.test_image_data_base64 = base64.b64encode(self.test_image_data).decode('utf-8')

        self.test_image2_path = 'main/tests/imgs/test_image_nx.jpg'
        self.test_image2_data = open(self.test_image2_path, 'rb').read()
        self.test_image2_data_base64 = base64.b64encode(self.test_image2_data).decode('utf-8')
        
        return super().setUp()

    def test_api_ios_get(self):
        response = self.client.get('/api/ios/image')
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, '"success": false')
        self.assertContains(response, '"description": "Request method must be POST"')
        self.assertContains(response, '"covid_percentage": 1')
        self.assertContains(response, '"normal_percentage": 1')
        self.assertContains(response, '"pneumonia_percentage": 1')
        self.assertContains(response, '"prediction": "null"')
        self.assertContains(response, '"image_url": "null"')

        response = self.client.get('/api/ios/image', {'image': self.test_image_data_base64})
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, '"success": false')
        self.assertContains(response, '"description": "Request method must be POST"')

    def test_api_ios_post(self):

        response = self.client.post('/api/ios/image')
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, '"success": false')
        self.assertContains(response, '"description": "No image found"')

        response = self.client.post('/api/ios/image', {'image': self.test_image_data_base64})
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, '"success": true')

        response = self.client.post('/api/ios/image', {'image': self.test_image2_data_base64})
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, '"success": false')
