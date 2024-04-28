# import packages
from torch.utils.data import DataLoader
from dataset import MIP, MIP_MNN
import torch
from sklearn.metrics import roc_auc_score, RocCurveDisplay
import matplotlib.pyplot as plt
import numpy as np
import pandas

# The method for calculating the AUC score was adapted from:
# Scikit-learn: Machine Learning in Python, Pedregosa et al., JMLR 12, pp. 2825-2830, 2011.

# evaluates loss and auc of model
def evaluate_state(model, dim, criterion,device):
    test_set = MIP(csv_file="./data/test_data.csv", dim=dim)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=True)
    true = []
    pred = []
    avg_loss = 0.0
    step = 0
    for img, label in test_loader:
        img = img.to(device)
        label = label.to(device)
        out = model(img)

        loss = criterion(out, label)
        avg_loss += loss.item()

        out = torch.sigmoid(out)

        out = out.cpu().detach().numpy()
        label = label.cpu().detach().numpy()
        pred.append(out[0,0])
        true.append(label[0,0])

        step += 1


    avg_loss /= step
    score = roc_auc_score(y_true = true, y_score=pred)

    # # OPTIONAL: visualize roc curve and save to roc_curve.png
    # display = RocCurveDisplay.from_predictions(y_true=true, y_pred=pred)
    # display.plot()
    # plt.title("Coronal Model")
    # plt.savefig("roc_curve_coronal.png")

    return  score, avg_loss

# plots a graph of losses and auc scores for each epoch, to visualize
# the improvement of the model throughout training
def evaluate_training(csv_file="./results/res_1_4.csv", ep=50):
    file = pandas.read_csv(csv_file)
    losses = np.zeros(ep, dtype="float32")
    aucs = np.zeros(ep, dtype="float32")
    epochs = np.zeros(ep, dtype="uint8")
    name = csv_file[-7:-4]

    for epoch in range(ep):
        losses[epoch] = file.iloc[epoch, 1]
        aucs[epoch] = file.iloc[epoch, 2]
        epochs[epoch] = epoch

    plt.title("loss and auc")
    plt.plot(epochs, losses, color="red")
    plt.plot(epochs, aucs, color="blue")
    plt.savefig(("loss_auc_{}.png").format(name))

# only used to test the multi model, not for evaluating model state during training
def evaluate_MNN(model, criterion, device):
    test_set = MIP_MNN(csv_file="./data/test_data.csv")
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)
    true = []
    pred = []
    avg_loss = 0.0
    step = 0
    for img0, img1, label in test_loader:
        img0 = img0.to(device)
        img1 = img1.to(device)
        label = label.to(device)
        out = model(img0, img1)

        loss = criterion(out, label)
        avg_loss += loss.item()

        out = torch.sigmoid(out)

        out = out.cpu().detach().numpy()
        label = label.cpu().detach().numpy()
        pred.append(out[0,0])
        true.append(label[0,0])

        step += 1


    avg_loss /= step
    score = roc_auc_score(y_true = true, y_score=pred)

    # # OPTIONAL: visualize roc curve and save to roc_curve.png
    # display = RocCurveDisplay.from_predictions(y_true=true, y_pred=pred)
    # display.plot()
    # plt.title("Multi Model")
    # plt.savefig("roc_curve_multi.png")

    return  score, avg_loss
