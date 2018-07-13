from django import forms
from django.contrib.auth.models import User

from django.contrib.auth.forms import UserCreationForm
from django.core.exceptions import ValidationError

class UserForm(forms.ModelForm):

	class Meta:
		model = User
		fields =  [
			'email',
			'password',			
		]

		labels = {
			'Email' : 'email',
			'Password': 'password',
		}
		widgets = {

			'email':forms.TextInput(attrs={'class':'form-control',
				'type' :'email', 
				'id':"email", 
				'required': True,	
				'placeholder':"email@example.com"}),
			'password':forms.PasswordInput(attrs={'class':'form-control',
				'id':"password",
				'placeholder' : "Password"

				}),


		}	

'''class RegistroForm(UserCreationForm):

	class Meta:
		model = User
		fields = [
				'username',
				'first_name',
				'last_name',
				'email',
				'password',
			]
		labels = {
				'username': 'Nombre de usuario',
				'first_name': 'Nombre',
				'last_name': 'Apellidos',
				'email': 'Correo',
				'password' : 'Password'
		}

'''










class CustomUserCreationForm(forms.Form):
    username = forms.CharField(label='Enter Username', min_length=4, max_length=150,widget = forms.TextInput(attrs={'class':'form-control'}))
    email = forms.EmailField(label='Enter email',widget = forms.TextInput(attrs={'class':'form-control'}))
    first_name = forms.CharField(label = 'Nombres',widget = forms.TextInput(attrs={'class':'form-control'}))
    last_name = forms.CharField(label = 'Apellidos',widget = forms.TextInput(attrs={'class':'form-control'}))
    password1 = forms.CharField(label='Enter password', widget=forms.PasswordInput(attrs={'class':'form-control'}))
    password2 = forms.CharField(label='Confirm password', widget=forms.PasswordInput(attrs={'class':'form-control'}))






    def clean_firts_name(self):
    	first_name = self.cleaned_data.get('first_name').lower()
    	return first_name
    def clean_last_name(self):
    	last_name = self.cleaned_data.get('last_name').lower()
    	return last_name
    def clean_username(self):
        username = self.cleaned_data['username'].lower()
        r = User.objects.filter(username=username)
        if r.count():
            raise  ValidationError("Username already exists")
        return username
    def clean_email(self):
        email = self.cleaned_data['email'].lower()
        r = User.objects.filter(email=email)
        if r.count():
      
           raise  ValidationError("Email already exists")
        return email
    def clean_password2(self):
        password1 = self.cleaned_data.get('password1')
        password2 = self.cleaned_data.get('password2')

        if password1 and password2 and password1 != password2:
            raise ValidationError("Password don't match")

        return password2

    def save(self, commit=True):
        user = User.objects.create_user(
            self.cleaned_data['username'],
            self.cleaned_data['email'],
            self.cleaned_data['password1'],
        )
        user.first_name= self.cleaned_data['first_name']
        user.last_name = self.cleaned_data['last_name']
        user.save()
        return user