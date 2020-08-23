import pandas as pd
from PIL import Image
import PIL.Image

PIL.Image.MAX_IMAGE_PIXELS = 47205375392

colSpecs = [(0, 37), (38, 300)]
inputData = pd.read_fwf('C:/Users/Roman/Downloads/Dataset 1 (Simplex)-20200416T113650Z-001/train.txt'
                        , colspecs=colSpecs, header=0)
for line in range(1, inputData.__len__() - 1):
    if inputData.loc[line].equals(inputData.loc[line - 1]):
        inputData.drop(line - 1, axis=0, inplace=True)

inputData.reset_index(drop=True, inplace=True)

write_path = 'C:/Users/Roman/Downloads/PotHole Dataset/train/hole/'
picIndex = 0
path_begin = 'C:/Users/Roman/Downloads/Dataset 1 (Simplex)-20200416T113650Z-001/'

for line in range(0, inputData.__len__() - 1):
    path = path_begin + inputData.loc[line][0].replace("\\", "/").replace(".bmp", ".JPG")
    holeData = (inputData.loc[line][1].split())
    numOfHoles = int(holeData[0])
    image = Image.open(path)
    for hole in range(numOfHoles):
        result = image.crop((int(holeData[1 + hole * 4]),
                             int(holeData[2 + hole * 4]),
                             int(holeData[3 + hole * 4]) + int(holeData[1 + hole * 4]),
                             int(holeData[4 + hole * 4]) + int(holeData[2 + hole * 4])))
        result.save(write_path + str(picIndex) + '.JPG')
        picIndex += 1
