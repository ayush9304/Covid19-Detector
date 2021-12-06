from django.test import TestCase, Client

# Create your tests here.

class TestMain(TestCase):
    def test_index(self):
        c = Client()
        response = c.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'main/index.html')
        self.assertContains(response, '<title>COVID-19 Detector | CoviScan</title>')
        self.assertContains(response, '<h5 class="card-header bg-light">COVID-19 Detection Using X-Rays</h5>')

    def test_api(self):
        c = Client()
        response = c.get('/api/docs')
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'main/api_docs.html')
        self.assertContains(response, '<title>API | CoviScan</title>')
        self.assertContains(response, '<h5 class="card-header bg-light">COVID-19 Detection APIs</h5>')

    def test_index_predict(self):
        c = Client()
        response = c.get('/predict')
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, '"success": false')
        self.assertContains(response, '"description": "Request method must be POST"')

    def test_api_warmup(self):
        c = Client()
        response = c.get('/warmup')
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, '"success": true')
        self.assertContains(response, '"description": "Warming up"')

    def test_api_image_get(self):
        c = Client()
        response = c.get('/api/image')
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, '"success": false')
        self.assertContains(response, '"description": "Request method must be POST"')

    def test_api_image_post(self):
        c = Client()
        response = c.post('/api/image')
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, '"success": false')
        self.assertContains(response, '"description": "No image found"')

    def test_api_ios_get(self):
        c = Client()
        response = c.get('/api/ios/image')
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, '"success": false')
        self.assertContains(response, '"description": "Request method must be POST"')
        self.assertContains(response, '"covid_percentage": 1')
        self.assertContains(response, '"normal_percentage": 1')
        self.assertContains(response, '"pneumonia_percentage": 1')
        self.assertContains(response, '"prediction": "null"')
        self.assertContains(response, '"image_url": "null"')

    def test_api_ios_post(self):
        c = Client()
        response = c.post('/api/ios/image')
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, '"success": false')
        self.assertContains(response, '"description": "No image found"')
        self.assertContains(response, '"covid_percentage": 1')
        self.assertContains(response, '"normal_percentage": 1')
        self.assertContains(response, '"pneumonia_percentage": 1')
        self.assertContains(response, '"prediction": "null"')
        self.assertContains(response, '"image_url": "null"')
    