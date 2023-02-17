from PIL import Image
from matplotlib.pyplot import imshow
import pandas
import matplotlib.pylab as plt
import os
import glob
import skillsnetwork
import torch

def show_data(data_sample, shape = (28, 28)):
    plt.imshow(data_sample[0].numpy().reshape(shape), cmap='gray')
    plt.title('y = ' + data_sample[1])
    
negative_file_path="./data/Negative"
os.listdir(negative_file_path)[0:3]
[os.path.join(negative_file_path,file) for file in  os.listdir(negative_file_path)][0:3]
negative_files=[os.path.join(negative_file_path,file) for file in  os.listdir(negative_file_path) if file.endswith(".jpg")]
negative_files.sort()
negative_files[0:3]

positive_file_path="./data/Positive"
os.listdir(positive_file_path)[0:3]
[os.path.join(positive_file_path,file) for file in  os.listdir(positive_file_path)][0:3]
positive_files=[os.path.join(positive_file_path,file) for file in  os.listdir(positive_file_path) if file.endswith(".jpg")]
positive_files.sort()
positive_files[0:3]

number_of_samples = len(positive_files + negative_files)

Y=torch.zeros([number_of_samples])
Y=Y.type(torch.LongTensor)
print(Y.type())

Y[::2]=1

Y[1::2]=0

all_files = [None] * number_of_samples
all_files[::2] = positive_files
all_files[1::2] = negative_files

for i in range(4):
    image = Image.open(all_files[i])
    imshow(image)
    print("Label: ", Y[i])
    plt.show()
    
train=False

if train:
    all_files=all_files[0:30000]
    Y=Y[0:30000]

else:
    all_files=all_files[30000:]
    Y=Y[30000:]