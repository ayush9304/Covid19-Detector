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

    def test_api_image_get(self):
        response = self.client.get('/api/image')
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, '"success": false')
        self.assertContains(response, '"description": "Request method must be POST"')

        response = self.client.get('/api/image', {'image': self.test_image_data, 'name': 'test_image2.png'})
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, '"success": false')
        self.assertContains(response, '"description": "Request method must be POST"')

    def test_api_image_post(self):
        response = self.client.post('/api/image')
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, '"success": false')
        self.assertContains(response, '"description": "No image found"')

        response = self.client.post('/api/image', {'image': self.test_image_data, 'name': 'test_image2.png'})
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, '"success": true')

        response = self.client.post('/api/image', {'image': self.test_image2_data, 'name': 'test_image_nx.png'})
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, '"success": false')
