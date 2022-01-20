from django.test import TestCase, Client

class BaseTest(TestCase):
    def setUp(self):
        self.client = Client()
        return super().setUp()

    def test_api_docs(self):
        response = self.client.get('/api/docs')
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, '<title>API | CoviScan</title>')
        self.assertContains(response, '<h5 class="card-header bg-light">COVID-19 Detection APIs</h5>')
        self.assertTemplateUsed(response,'main/api_docs.html')
