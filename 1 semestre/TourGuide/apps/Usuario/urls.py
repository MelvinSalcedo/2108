from django.conf.urls import url, include

from apps.Usuario.views import index,user_login,User_index_view,info,logout_view,Historial,view_information,view_Profile,register
from django.urls import path


app_name = 'apps'
urlpatterns=[

	path('Home/',index, name = "index"),
	path('Login/',user_login,name = "login"),
	path('User/',User_index_view,name='User_index_view'),
	path('Result/',info, name = 'rresult'),
	path('',logout_view, name = 'logout'),
	path('Record/',Historial, name = 'Historial'),
	path('<int:imagen_id>/',view_information, name = 'informacion'),
	path('Profile/<slug:id>/',view_Profile,name='profile'),
	path('reg_form/',register,name='register')


]