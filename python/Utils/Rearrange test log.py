import pandas as pd

log_path = 'C:/Users/Roman/Documents/Python/Project/Data/PotHole Dataset/models MobileNetV2/'
log_name = 'ALL test log.txt'

result_file = open(log_path + log_name + ' Rearranged.txt', "w")
inputData = pd.read_fwf(log_path + log_name)

result_list = []

for line in range(1, inputData.__len__() - 1):
    result_list.append([str(inputData.loc[line][0]) + '\n',
                        float(inputData.loc[line][0].split('Hole Detection Chance: ')[1].split(' ')[0]),
                        float(inputData.loc[line][0].split('Non-Hole Detection Chance: ')[1])])

# buble-sort the models by higher chances
temp_list = []
for x in range(len(result_list)):
    for i in range(len(result_list) - 1):
        if result_list[i + 1][1] + result_list[i + 1][2] > result_list[i][1] + result_list[i][2]:
            temp_list.append(result_list[i])
            result_list[i] = result_list[i + 1]
            result_list[i + 1] = temp_list.pop()

for i in result_list:
    result_file.write(i[0])
result_file.close()
