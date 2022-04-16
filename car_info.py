import os

TARGET = 'model'

files = [file for file in os.listdir('./combinations/class') if file.endswith(".jpg")]
file_paths = ['./combinations/class' + file for file in files]

if TARGET == 'make':
    classes = list(set([file.split('_')[0] for file in files]))
if TARGET == 'model':
    classes = list(set([file.split('_')[0] + '_' + file.split('_')[1] for file in files]))
print()
print()
print()
print(classes)