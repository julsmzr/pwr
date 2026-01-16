import os

from cv2 import resize
import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision.models import resnet18, ResNet18_Weights


classes=["cat", "horse"]
X = []
y = []

for class_id, cls in enumerate(classes):
    files = os.listdir(f"datasets/animals-10/{cls}")
    for file_id, filename in enumerate(files[:100]):
        img_path = f"datasets/animals-10/{cls}/{filename}"
        img = plt.imread(img_path)

        # take care of images (png) with different amount of color channels
        if img.shape[2] == 3:
            X.append(resize(img, (224, 224)))
            y.append(class_id)
        
X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train = torch.from_numpy(np.moveaxis(X_train, 3, 1)).float() # move channels to be first using moveaxis: channels x height x width
X_test = torch.from_numpy(np.moveaxis(X_test, 3, 1)).float() 

y_train = torch.from_numpy(y_train).long() 
y_test = torch.from_numpy(y_test).long() 


# Model
batch_size = 8
n_epochs = 100
weights = None
# weights = ResNet18_Weights.IMAGENET1K_V1
model = resnet18(weights=weights)
# print(model)

# change the clf head
num_features = model.fc.in_features
model.fc = nn.Linear(in_features=num_features, out_features=1) # 1 for binary

device = torch.device("mps")
model = model.to(device)

optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
criterion = nn.BCEWithLogitsLoss()

train_dataset = TensorDataset(X_train, y_train)
train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# fit
from tqdm import tqdm


for epoch in tqdm(range(n_epochs)):

    model.train() # training mode
    for i, batch in enumerate(train_data_loader):
        inputs, labels = batch
        optimizer.zero_grad()
        outputs = model(inputs.to(device))
        loss = criterion(outputs.to(device), labels.unsqueeze(1).float().to(device))

        # backprop
        loss.backward()
        optimizer.step()

    model.eval()
    logits = model(X_test.to(device))
    # print(logits) # if neg class 0 if pos class 1
    y_pred = (logits.cpu().detach().numpy() > 0).astype(int)
    score = balanced_accuracy_score(y_test, y_pred)
    print("epoch", epoch, "score:", score)




