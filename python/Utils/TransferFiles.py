from glob import glob
from PIL import Image
import random
import shutil

data_path0 = 'C:/Users/Roman/Documents/Python/Project/Data/lineSet Right Data/0/'
write_path_train0 = 'C:/Users/Roman/Documents/Python/Project/Data/lineSet Right/train/0'
write_path_test0 = 'C:/Users/Roman/Documents/Python/Project/Data/lineSet Right/test/0'
write_path_eval0 = 'C:/Users/Roman/Documents/Python/Project/Data/lineSet Right/eval/0'
image_files0 = glob(data_path0 + '*')

data_path1 = 'C:/Users/Roman/Documents/Python/Project/Data/lineSet Right Data/1/'
write_path_train1 = 'C:/Users/Roman/Documents/Python/Project/Data/lineSet Right/train/1'
write_path_test1 = 'C:/Users/Roman/Documents/Python/Project/Data/lineSet Right/test/1'
write_path_eval1 = 'C:/Users/Roman/Documents/Python/Project/Data/lineSet Right/eval/1'
image_files1 = glob(data_path1 + '*')

for e in image_files0:
    x = random.randrange(7)
    if x == 0:
        shutil.move(e, write_path_train0)
        continue
    if x == 1:
        shutil.move(e, write_path_train0)
        continue
    if x == 2:
        shutil.move(e, write_path_train0)
        continue
    if x == 3:
        shutil.move(e, write_path_train0)
        continue
    if x == 4:
        shutil.move(e, write_path_train0)
        continue
    if x == 5:
        shutil.move(e, write_path_test0)
        continue
    if x == 6:
        shutil.move(e, write_path_eval0)
        continue

for e in image_files1:
    x = random.randrange(7)
    if x == 0:
        shutil.move(e, write_path_train1)
        continue
    if x == 1:
        shutil.move(e, write_path_train1)
        continue
    if x == 2:
        shutil.move(e, write_path_train1)
        continue
    if x == 3:
        shutil.move(e, write_path_train1)
        continue
    if x == 4:
        shutil.move(e, write_path_train1)
        continue
    if x == 5:
        shutil.move(e, write_path_test1)
        continue
    if x == 6:
        shutil.move(e, write_path_eval1)
        continue

