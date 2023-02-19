from PIL import Image
from matplotlib.pyplot import imshow
import matplotlib.pylab as plt
import os
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torchvision import transforms, models

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
            transforms.Resize((224, 224)),
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

class ResNet18Model(nn.Module):

    def __init__(self):
        super(ResNet18Model, self).__init__()
        self.resnet18 = models.resnet18(pretrained=True)
        num_ftrs = self.resnet18.fc.in_features
        self.resnet18.fc = nn.Linear(num_ftrs, 10)

    def forward(self, x):
        x = self.resnet18(x)
        return nn.functional.softmax(x, dim=1)
        
model = ResNet18Model()
    
torch.manual_seed(0)

learning_rate = 0.1
momentum = 0.1
batch_size = 5
loss_fn = nn.CrossEntropyLoss()

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

model = ResNet18Model()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)

num_epochs = 5
best_val_acc = 0.0
train_losses, train_accs, val_losses, val_accs = [], [], [], []

for epoch in range(num_epochs):
    print("start train")
    model.train()
    train_loss = 0.0
    train_correct = 0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1) # define preds variable here
        train_correct += torch.sum(preds == labels.data)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * inputs.size(0)

    train_loss = train_loss / len(train_loader.dataset)
    train_acc = train_correct.double() / len(train_loader.dataset)
    print("train loss: {:.4f}; accuracy: {:.4f}".format(train_loss, train_acc))

    model.eval()
    val_loss = 0.0
    val_correct = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)

            val_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            val_correct += torch.sum(preds == labels.data)

    val_loss = val_loss / len(val_loader.dataset)
    val_acc = val_correct.double() / len(val_loader.dataset)
    print("val loss: {:.4f}; accuracy: {:.4f}".format(val_loss, val_acc))

    if val_acc > best_val_acc:
        best_val_acc = val_acc

print("Training complete. Best validation accuracy: {:.4f}".format(best_val_acc))

# Plot the accuracy and loss
accuracy = [train_acc, val_acc]
loss = [train_loss, val_loss]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

ax1.plot(accuracy)
ax1.set_title('Accuracy')
ax1.set_xticks([0, 1])
ax1.set_xticklabels(['Train', 'Val'])
ax1.set_ylim([0, 1])

ax2.plot(loss)
ax2.set_title('Loss')
ax2.set_xticks([0, 1])
ax2.set_xticklabels(['Train', 'Val'])

plt.savefig('accuracy_loss.png')

# Show the plot
plt.show()