from glob import glob
import time
import os
from random import random

path = 'C:/Users/Roman/Documents/Python/Project/Data/Japan holes'
image_files = glob(path + '/*/*.jpg')

for e in image_files:
    path_no_name = e.split('\\')[0] + '/' + e.split('\\')[-2] + '/'
    name = str(time.time()) + str(random()) + '.JPG'
    # print(e)
    # print(path_no_name + name)
    os.rename(e, path_no_name + name)
