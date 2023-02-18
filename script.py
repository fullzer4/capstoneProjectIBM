from PIL import Image
from matplotlib.pyplot import imshow
import matplotlib.pylab as plt
import os
import torch
from torch.utils.data import Dataset
import torch.nn as nn
from torchvision import transforms

class CustomDataset(Dataset):

    def __init__(self, train=True):
        negative_file_path = "./data/Negative"
        negative_files = [os.path.join(negative_file_path, file) for file in os.listdir(negative_file_path) if file.endswith(".jpg")]
        negative_files.sort()

        positive_file_path = "./data/Positive"
        positive_files = [os.path.join(positive_file_path, file) for file in os.listdir(positive_file_path) if file.endswith(".jpg")]
        positive_files.sort()

        number_of_samples = len(positive_files + negative_files)

        Y = torch.zeros([number_of_samples])
        Y = Y.type(torch.LongTensor)

        Y[::2] = 1
        Y[1::2] = 0

        all_files = [None] * number_of_samples
        all_files[::2] = positive_files
        all_files[1::2] = negative_files

        if train:
            self.all_files = all_files[:30000]
            self.Y = Y[:30000]
        else:
            self.all_files = all_files[30000:]
            self.Y = Y[30000:]

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx):
        image = Image.open(self.all_files[idx])
        image = self.transform(image)
        Y = self.Y[idx]
        return image, Y

train_dataset = CustomDataset(train=True)
val_dataset = CustomDataset(train=False)

image, Y = val_dataset[15]
imshow(image.permute(1,2,0))
print("Label: ", Y)
plt.show()

image, Y = val_dataset[102]
imshow(image.permute(1,2,0))
print("Label: ", Y)
plt.show()

class SoftmaxModel(nn.Module):

    def __init__(self, size_of_image):
        super(SoftmaxModel, self).__init__()
        self.size_of_image = size_of_image
        self.linear = nn.Linear(size_of_image, 2)

    def forward(self, x):
        x = x.view(-1, self.size_of_image)
        x = self.linear(x)
        return nn.functional.softmax(x, dim=1)
    
torch.manual_seed(0)

learning_rate = 0.1
momentum = 0.1
batch_size = 5
loss_fn = nn.CrossEntropyLoss()

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

model = SoftmaxModel(size_of_image=3*224*224)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

num_epochs = 5
best_val_acc = 0.0

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    train_correct = 0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        train_correct += torch.sum(preds == labels)

    train_loss = train_loss / len(train_dataset)
    train_acc = train_correct.double() / len(train_dataset)

    model.eval()
    val_loss = 0.0
    val_correct = 0
    for inputs, labels in val_loader:
        outputs = model(inputs)
        loss = loss_fn(outputs, labels)

        val_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        val_correct += torch.sum(preds == labels)

    val_loss = val_loss / len(val_dataset)
    val_acc = val_correct.double() / len(val_dataset)

    print('Epoch [{}/{}], Train Loss: {:.4f}, Train Acc: {:.4f}, Val Loss: {:.4f}, Val Acc: {:.4f}'.format(
        epoch+1, num_epochs, train_loss, train_acc, val_loss, val_acc))

    if val_acc > best_val_acc:
        best_val_acc = val_acc

print('Best Validation Accuracy: {:.4f}'.format(best_val_acc))