import numpy as np
import neurolab as nl

import random
from PIL import Image
import os
import re

#comentario de prueba de brunch
"""# **Convierte una matriz x en vector**"""

def mat2vec(x):
    m = x.shape[0]*x.shape[1]
    tmp1 = np.zeros(m)

    c = 0
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            tmp1[c] = x[i,j]
            c +=1
    return tmp1

def w_peso (x):
  

"""
# **Lee un archivo de imagen y lo convierte a un arreglo Numpy**"""

def readImg2array(file,size, threshold= 145):
    pilIN = Image.open(file).convert(mode="L")
    pilIN= pilIN.resize(size)
    #pilIN.thumbnail(size,Image.ANTIALIAS)
    imgArray = np.asarray(pilIN,dtype=np.uint8)
    x = np.zeros(imgArray.shape,dtype=float)
    x[imgArray > threshold] = 1
    x[x==0] = -1
    return x

"""# **Arreglo a imagen**"""

def array2img(data, outFile = None):
    #data is 1 or -1 matrix
    y = np.zeros(data.shape,dtype=np.uint8)
    y[data==1] = 255
    y[data==-1] = 0
    img = Image.fromarray(y,mode="L")
    if outFile is not None:
        img.save(outFile)
    return img



"""# **Se carga un archivo de imagen para el entrenamiento**"""

#Main
#First, you can create a list of input file path
current_path = os.getcwd()
train_paths = []
target = []
path = current_path+"/train_pics/"
for i in os.listdir(path):
    
    if re.match(r'..jpg',i):
        train_paths.append(path+i)
        #read image and convert it to Numpy array
        print(path+i)
        print("Importando Imágenes....")
  

size=(100,100)
threshold=60
num_files = 0
path=train_paths[0]

x = readImg2array(file=path,size=size,threshold=threshold)
x_vec = mat2vec(x)
target = np.array(x_vec,ndmin=2)

'''
for path in train_paths:
  x = readImg2array(file=path,size=size,threshold=threshold)
  x_vec = mat2vec(x)
  target.append(np.array(x_vec,ndmin=2))
print("El vector de entrada está creado!!")
print(len(target), target)'''

current_path = os.getcwd()
test_paths = []
path = current_path+"/test_pics/"
for i in os.listdir(path):
  if re.match(r'[0-9a-zA-Z-]*.jpg',i):
    test_paths.append(path+i)
    #read image and convert it to Numpy array
print("Importando imágenes....")


size=(100,100)
threshold=60

#num_files is the number of files
counter = 0
for path in test_paths:
  print(path)
  y = readImg2array(file=path,size=size,threshold=threshold)
  oshape = y.shape
  y_img = array2img(y)
  y_img.show()
  y_vec = mat2vec(y)
  test = np.array(y_vec,ndmin=2)
  print("Datos de Test importados!")

print("El vector de test está creado!!")
print(test)

"""# **No utilizado aún**"""

#print("Updating...")
#y_vec_after = update(w=w,y_vec=y_vec,theta=theta,time=time)
#y_vec_after = y_vec_after.reshape(oshape)
#if current_path is not None:
#  outfile = current_path+"/after_"+str(counter)+".jpeg"
#  array2img(y_vec_after,outFile=outfile)
#else:
#  after_img = array2img(y_vec_after,outFile=None)
#  after_img.show()
#  counter +=1

"""# **Crea una Red de Hopfielf para el vector de entrada target**"""

# Create and train network
net = nl.net.newhop(target,transf= nl.trans.HardLims())

"""# **Prueba el vector de entrada target**
revisar y corregir

"""

output_target = net.sim(target)
print(output_target.shape)
print(target.shape)
print(test.shape)
output_test = net.sim(test)

for i in range(len(target)):
  print((output_target[i] == target[i]).all(), 'Sim. steps',len(net.layers[0].outs))

for i in range(len(test)):
  print((output_test[i] == test[i]).all(), 'Sim. steps',len(net.layers[0].outs))



after_img = array2img(output_test,outFile='salida.jpg')