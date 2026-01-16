import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold

# Showing how to work with an image dataset

import matplotlib.pyplot as plt
import numpy as np
import os
from cv2 import resize

from torch.utils.data import DataLoader, TensorDataset
from torchvision.models import resnet18, ResNet18_Weights

from torch import optim

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score

from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold


import torch.nn as nn
import torch

from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names

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
# weights = None
weights = ResNet18_Weights.IMAGENET1K_V1 # pretrained on imagenet
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

scores = np.zeros((n_epochs))
for epoch in tqdm(range(n_epochs)):
    fig, ax = plt.subplots(1, 1, figsize=(15, 5))
    fig, ax = plt.subplot_mosaic(["a", "a", "b"])

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
    scores[epoch] = balanced_accuracy_score(y_test, y_pred)

    ax["a"].plot(range(epoch + 1), scores[:epoch+1])


    # n = get_graph_node_names(model)
    # use the model as feature extractor

    return_nodes = {
        "flatten": "flatten"
    }

    # basically pca but its not a pca
    extractor = create_feature_extractor(model, return_nodes = return_nodes)
    X_extracted = extractor(X_train.to(device))["flatten"].cpu().detach().numpy() # this is the input to the last clf layer same thing we did here     # X = X.reshape((X.shape[0], X.shape[1] * X.shape[2] * X.shape[3])) # flatten
    pca = PCA(n_components=2).fit(X_extracted)
    
    clf = LogisticRegression().fit(X_extracted, y_train)
    score = balanced_accuracy_score(y_test, clf.predict(X_test))

    X_extracted = extractor(X_test.to(device))["flatten"].cpu().detach().numpy() # this is the input to the last clf layer same thing we did here     # X = X.reshape((X.shape[0], X.shape[1] * X.shape[2] * X.shape[3])) # flatten
    X_extracted_plot = pca.transform(X_extracted)

    # pca feature space
    ax["b"].scatter(X_extracted_plot[:, 0], X_extracted_plot[:, 1], c=y_test, cmap="bwr_r")


    plt.tight_layout()
    plt.savefig("foo.png", dpi=300)
    plt.close()




