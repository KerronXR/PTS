import pandas as pd
from PIL import Image
import PIL.Image
from glob import glob

PIL.Image.MAX_IMAGE_PIXELS = 47205375392

dataPath = 'C:/Users/Roman/Downloads/RDD2020_data.tar/RDD2020_data/xmls/'
imagesPath = 'C:/Users/Roman/Downloads/RDD2020_data.tar/RDD2020_data/images/'
writePath = 'C:/Users/Roman/Downloads/RDD2020_data.tar/RDD2020_data/holes/'
imagePath = ''
dataSet = glob(dataPath + '*.xml')
picIndex = 0

for d in range(4250, len(dataSet)):
    inputData = pd.read_fwf(dataSet[d])
    for line in range(1, inputData.__len__() - 1):
        if inputData.loc[line][0].split('<')[1].split('>')[0] == 'filename':
            imagePath = imagesPath + inputData.loc[line][0].split('<')[1].split('>')[1]
            print(imagePath)
        if inputData.loc[line][0].split('<')[1].split('>')[0] == 'object':
            subfolder = inputData.loc[line + 1][0].split('<')[1].split('>')[1]
            if inputData.loc[line + 2][0].split('<')[1].split('>')[0] == 'bndbox':
                xmin = inputData.loc[line + 3][0].split('<')[1].split('>')[1]
                ymin = inputData.loc[line + 4][0].split('<')[1].split('>')[1]
                xmax = inputData.loc[line + 5][0].split('<')[1].split('>')[1]
                ymax = inputData.loc[line + 6][0].split('<')[1].split('>')[1]
            else:
                xmin = inputData.loc[line + 6][0].split('<')[1].split('>')[1]
                ymin = inputData.loc[line + 7][0].split('<')[1].split('>')[1]
                xmax = inputData.loc[line + 8][0].split('<')[1].split('>')[1]
                ymax = inputData.loc[line + 9][0].split('<')[1].split('>')[1]
            image = Image.open(imagePath)
            result = image.crop((int(xmin),
                                 int(ymin),
                                 int(xmax),
                                 int(ymax)))
            result.save(writePath + subfolder + '/' + str(picIndex) + '.JPG')
            picIndex += 1
