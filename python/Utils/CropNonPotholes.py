from glob import glob
from PIL import Image
import random
import PIL.Image

PIL.Image.MAX_IMAGE_PIXELS = 47205375392

data_path = 'C:/Users/Roman/Downloads/Negative data/'
write_path = 'C:/Users/Roman/Downloads/PotHole Dataset/train/not hole/'
image_files = glob(data_path + '*.jpg')

x_pos = 1547
y_pos = 1435
width = 64
height = 32
picIndex = 0

for e in image_files:
    image = Image.open(e)
    x = x_pos + random.randint(0, 898)
    y = y_pos + random.randint(0, 188)
    w = x + width + random.randint(0, 128)
    h = y + height + random.randint(0, 64)
    result = image.crop((x, y, w, h))
    result.save(write_path + str(picIndex) + '.JPG')
    picIndex += 1
