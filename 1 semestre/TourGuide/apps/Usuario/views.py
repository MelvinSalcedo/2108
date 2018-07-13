
from django.shortcuts import render, redirect
from django.http import HttpResponse
from django.contrib.auth import login, authenticate,logout
from apps.Usuario.forms import UserForm,CustomUserCreationForm
from django.contrib.auth.forms import UserCreationForm
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.views.generic import CreateView

from django.core.files.storage import FileSystemStorage
from django.core.files.storage import Storage
from django.core.files.base import ContentFile
from django.contrib.auth.models import User
from apps.Usuario.models import ImagenInfo,HistorialUsuario
from django.urls import reverse_lazy


#Api Google
import io
import os

# Imports the Google Cloud client library
from google.cloud import vision
from google.cloud.vision import types

# Create your views here.




def index(request):
	
	return render(request,'Home/Home.html')

#--------------------------------------------------------
#Solucion opcional al problema del login CSRF verification failed. Request aborted
@login_required
def User_index_view(request):
 
	return render(request,'Usuario/User_Index.html')
#---------------------------------------------------------


#Funcione de autentificacion mediante Email and password
#-----------------------------------------------------------
def user_login(request):
	
	if request.method == 'POST':
		form =  UserForm(request.POST)
		if form.is_valid():
			user = authenticate(
				username = request.POST['email'],
				password = request.POST['password'] 
			)
			if user is not None:
				login(request,user)
				contexto = {'usuarios': user}
				return redirect('login:User_index_view')
			else:
				messages.success(request,"Incorrect user or password")
				
	else:
			form = UserForm()
	return render(request,'Login/Login.html',{'form':form})
#---------------------------------------------------------------------------	

#Funcion de envio y respuesta API Google Vision
@login_required
def info(request):

	os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "melvin-afa24119e798.json"	
	vision_client = vision.ImageAnnotatorClient()
	myfile = request.FILES["file1"]	
	fs = FileSystemStorage()
	filename = fs.save(myfile.name, myfile)
	file_name = fs.url(filename)

	
	with io.open(file_name,"rb") as image_file:
		content = image_file.read()
		image = types.Image(content=content)
	fs.delete(filename)
	response = vision_client.web_detection(image = image)
	#response2 = vision_client.web_detection(image = image)
	annotations = response.web_detection

	label_data =""
	if annotations.web_entities:	
			c = 0
			print('Labels: ')
			for label in annotations.web_entities:
				if(c==0):

					label_data = label_data + label.description
				else:
					break
				c = c+1
	token = ImagenInfo.objects.filter(nombreimagen=label_data).exists()
	if token == True:

		token2 = ImagenInfo.objects.filter(nombreimagen=label_data)
		token3 = User.objects.filter(id= request.user.id)
		
		record =  HistorialUsuario(user = token3[0] ,ImagenInfo=token2[0])
		record.save()

		return render(request,'Usuario/result.html',{"labels":label_data , "image":annotations,"informacion" : token2})
	else:
		
		return render(request,'Usuario/result.html',{"labels":label_data , "image":annotations})

#------------------------------------------------------------------------------------------------------------
#Funcion que consulta y muestra de Historuial del usuario, necesita un User para poder acceder

@login_required
def Historial(request):
	Qhistorial = HistorialUsuario.objects.filter(user = request.user.id )
	contexto = {'objetos' : Qhistorial}
	return render(request,'Usuario/historial.html',contexto)
#Funcion de Ver informacion de la imagen del historial necesita como principal valor el  id de la imagen que se desea observar

@login_required
def view_information(request, imagen_id):
	imagen = ImagenInfo.objects.get(idimagen = imagen_id)
	contexto = {'imagen':imagen}
	return render(request,'Usuario/InfoImagen.html',contexto)

@login_required
def  view_Profile(request,id):
	return render(request,'Usuario/Profile.html')



def register(request):
    if request.method == 'POST':
        f = CustomUserCreationForm(request.POST)
        if f.is_valid():
            f.save()
            messages.success(request, 'Account created successfully')
            return redirect('login:login')

    else:
        f = CustomUserCreationForm()

    return render(request, 'Usuario/reg_form.html', {'form': f})

'''class RegistroUsuario(CreateView):
	model = User
	template_name = "Usuario/reg_form.html"
	form_class = RegistroForm
	success_url = reverse_lazy('login:login')'''





# Funcion que cierra (elimin la session creada por login(request,user)) a
def logout_view(request):
	logout(request)
	return redirect('index')