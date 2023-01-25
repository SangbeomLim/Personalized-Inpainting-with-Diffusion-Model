import os
from tqdm.auto import tqdm

file_path = "/home/user/Paint-by-Example/dataset/ohouse_images/annotations"

file_list = os.listdir(file_path)
print(len(file_list))

total_class_ratio = {}

bbox_list = []
label_list = []
for file in tqdm(file_list):
    file = os.path.join(file_path, file)
    with open(file) as f:
        line = f.readline()
        while line:
            line_split = line.strip('\n').split(" ")
            try:
                class_label = line_split[4]  # Label 값 가져오기
            except:
                print(line_split)
                # input()
            if class_label not in total_class_ratio:
                total_class_ratio[class_label] = 1
            else:
                total_class_ratio[class_label] += 1
            line = f.readline()  # 다음 행

d1 = {k: round(total_class_ratio[k]/sum(total_class_ratio[k] for k in total_class_ratio),2) for k in total_class_ratio}
print(total_class_ratio)
print(d1)
