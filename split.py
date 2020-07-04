import os
import shutil
import numpy as np

path = '/home/harsh/Downloads/ATAT/'

list = os.listdir(path)
np.random.shuffle(list)

train,val,test = np.split(list,[32,36])

for folder in os.listdir(path):
    if folder in train:
        shutil.copytree(path+folder,path+'train/'+folder)
    elif folder in val:
        shutil.copytree(path+folder,path+'val/'+folder)
    else:
        shutil.copytree(path+folder,path+'test/'+folder)
