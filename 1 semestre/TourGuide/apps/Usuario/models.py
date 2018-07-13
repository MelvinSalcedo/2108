from django.db import models
from django.contrib.auth.models import User
from datetime import date

# Create your models here.


class ImagenInfo(models.Model):
    idimagen = models.AutoField( primary_key=True) 
    nombreimagen = models.CharField( max_length=45)  
    ciudad = models.CharField( max_length=15)  
    provincia = models.CharField( max_length=15)  
    distrito = models.CharField( max_length=15)  
    direccion = models.CharField( max_length=15)  
    longitud = models.CharField(max_length=45)
    latitud = models.CharField(max_length=45)
    description = models.TextField(max_length=2000)
    construccion = models.CharField( max_length=20, blank=True, null=True)  
    estilo_arquitectonico = models.CharField( max_length=20, blank=True, null=True)  

class HistorialUsuario(models.Model):
    #idhistorial_usuario = models.AutoField(db_column='idHistorial_Usuario', primary_key=True)  # Field name made lowercase.
    user = models.ForeignKey(User, null = True , blank = True , on_delete = models.CASCADE) 
    ImagenInfo = models.ForeignKey(ImagenInfo, null = True , blank = True , on_delete = models.CASCADE)  
    Register_date = models.DateTimeField(auto_now_add=True)
