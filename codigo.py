import numpy as np
import neurolab as nl
import random
from PIL import Image
import os
import re

def mat2vec(x):
    m = x.shape[0]*x.shape[1]
    tmp1 = np.zeros(m)

    c = 0
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            tmp1[c] = x[i,j]
            c +=1
    return tmp1

def readImg2array(file,size, threshold= 145):
    pilIN = Image.open(file).convert(mode="L")
    pilIN= pilIN.resize(size)
    #pilIN.thumbnail(size,Image.ANTIALIAS)
    imgArray = np.asarray(pilIN,dtype=np.uint8)
    x = np.zeros(imgArray.shape,dtype=float)
    x[imgArray > threshold] = 1
    x[x==0] = -1
    return x

def array2img(data, outFile = None):
    #data is 1 or -1 matrix
    y = np.zeros(data.shape,dtype=np.uint8)
    y[data==1] = 255
    y[data==-1] = 0
    img = Image.fromarray(y,mode="L")
    if outFile is not None:
        img.save(outFile)
    return img

current_path = os.getcwd()
path = current_path+"/train_pics/"
train_paths = []
names = []

for i in os.listdir(path):
    if re.match(r'..jpg',i):
        train_paths.append(path+i)

for path in train_paths:
    name = path.split("/")[-1].split(".")[0]
    names.append(name)

train_paths

size = (100,100)
threshold = 60
train = []

for path in train_paths:
    x = readImg2array(path,size,threshold)
    x = mat2vec(x)
    train.append(x)

train = np.array(train)

net = nl.net.newhop(train)
#,transf= nl.trans.HardLims()

noise_percent = 0.1
test = []
rango = range(0,size[0]*size[1])
noise_level = round((rango[-1]+1)*noise_percent)

for i in range (len(train)):
    noise = random.sample(rango, noise_level)
    image_noisy = train[i].copy()
    image_noisy[noise] = -image_noisy[noise]
    test.append(image_noisy)
    array2img(image_noisy.reshape(size),outFile = train_paths[i].replace("train","test"))

test = np.array(test)

output_train = net.sim(train)

for i in range(len(output_train)):
    print(names[i], ": ", (output_train[i] == train[i]).all(), 'Sim. steps', len(net.layers[0].outs))
    array2img(output_train[i].reshape(size),outFile = "./output_pics/"+names[i]+"_train.jpg")
output_test = net.sim(test)

for i in range(len(output_train)):
    print(names[i],": ",(output_test[i] == train[i]).all(), 'Sim. steps', len(net.layers[0].outs))
    array2img(output_test[i].reshape(size),outFile = "./output_pics/"+names[i]+"_test.jpg")

