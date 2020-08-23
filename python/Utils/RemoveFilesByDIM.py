import os
from PIL import Image
import PIL.Image
from glob import glob
import numpy as np

path = 'C:/Users/Roman/Documents/Python/Project/Data/BigSet'
image_files = glob(path + '/*/*.jpg')
to_remove = []

for e in image_files:
    image = Image.open(e)
    width, height = image.size
    if width + height < 100:
        to_remove.append(e)
        print(width, height)

for e in to_remove:
    os.remove(e)
