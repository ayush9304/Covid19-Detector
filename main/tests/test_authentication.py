from django.test import TestCase, Client

class BaseTest(TestCase):
    def setUp(self):
        self.client = Client()
        self.register_url = '/register'
        self.login_url = '/login'
        
        self.user={
            'email':'testemail@mail.com',
            'username':'username',
            'password':'password',
            'confirmation':'password',
            'firstname':'fname',
            'lastname':'lname',
        }
        self.user_unmatching_password={

            'email':'testemail@gmail.com',
            'username':'username',
            'password':'teslatt',
            'confirmation':'teslatto',
            'firstname':'fname',
            'lastname': 'lname'
        }

        return super().setUp()

    
class RegisterTest(BaseTest):
    def test_can_view_page_correctly(self):
        response = self.client.get(self.register_url)
        self.assertEqual(response.status_code,200)
        self.assertTemplateUsed(response,'main/register.html')

    def test_can_register_user(self):
        response = self.client.post(self.register_url,self.user)
        self.assertEqual(response.status_code,200)

    def test_cant_register_user_with_unmatching_passwords(self):
        response = self.client.post(self.register_url,self.user_unmatching_password)
        self.assertEqual(response.status_code,200)
        self.assertContains(response, '<span class="star small" id="testboxmessage">Passwords must match.</span>')

    def test_cant_register_user_with_taken_username(self):
        self.client.post(self.register_url,self.user)
        response = self.client.post(self.register_url,self.user)
        self.assertEqual(response.status_code,200)
        self.assertContains(response, '<span class="star small" id="testboxmessage">Username already taken.</span>')
        
    def test_can_view_login_page(self):
        response = self.client.get(self.login_url)
        self.assertEqual(response.status_code,200)
        self.assertTemplateUsed(response,'main/login.html')
