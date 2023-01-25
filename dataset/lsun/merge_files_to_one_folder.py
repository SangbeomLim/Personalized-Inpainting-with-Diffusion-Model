import os

dir_path = "/home/user/Paint-by-Example/dataset/lsun/raw"

for (root, directories, files) in os.walk(dir_path):
    for file in files:
        if 'jpg' in file:
            file_path = os.path.join(root, file)
            new_file_path=os.path.join(dir_path,file)
            os.rename(file_path,new_file_path)

# train, valid, test= 240000,30000,30000
# for (root, directories, files) in os.walk(dir_path):
#     if train !=0:
#         for file in files:
#            file_path = os.path.join(root, file)