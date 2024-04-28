# import packages
import torch
from dataset import MIP
from resnet import ResNet
import torchvision.transforms.v2 as transforms
from torch.utils.data import DataLoader
from evaluate import evaluate_state
import csv
print("Packages loaded.")

# train model, arguments are hyperparameters
def train(dim=1, 
        batch_size=4, 
        epochs=50, 
        learning_rate=1e-5, 
        criterion=torch.nn.BCEWithLogitsLoss(), 
        data_transforms=transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=30),
            ]),
        device=torch.device("cuda:0" if torch.cuda.is_available else "cpu"),
        data_path="./data/train_data.csv"
        ):

    print("Using {}.".format(device))

    # create stats
    epoch_losses = []
    epoch_auc = []
    best_auc_model = "axial/model_1_2_3"
    best_auc = 0
    best_loss_model =  "axial/model_1_2_3"
    best_loss = 1

    # create ResNet34 model
    model=ResNet()
    model.to(device)

    # set optimization function
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # train for epochs
    for epoch in range(epochs):

        # create data splits
        train_set = MIP(csv_file=data_path, dim=dim, transform=data_transforms)
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

        for image, label in train_loader:
            optimizer.zero_grad()
            image = image.to(device)
            label = label.to(device)

            out = model(image)
            loss = criterion(out, label)

            loss.backward()
            optimizer.step()
        
        # calculate epoch auc and loss
        score, epoch_loss = evaluate_state(model, dim, criterion, device)
        epoch_losses.append(epoch_loss)
        epoch_auc.append(score)

        # save epoch results
        model_name = ("mip_{}_{}").format(batch_size, epoch + 1)
        model_type = "coronal" if dim else "sagittal"
        model_id = ("{}/{}").format(model_type, model_name)

        if score >= 0.9:
            torch.save(model.state_dict(), ("./training_models/{}.pth").format(model_id))

        if (score > best_auc):
            best_auc = score
            best_auc_model = model_id
        
        if (epoch_loss < best_loss):
            best_loss = epoch_loss
            best_loss_model = model_id

    # save training results
    csv_file = ("./training_results/res_{}_{}.csv").format(dim, batch_size)
    with open(csv_file, 'w', newline='') as res:
        writer = csv.writer(res)
        writer.writerow(["epoch", "loss", "auc"])
        for epoch in range(epochs):
            writer.writerow([epoch+1, epoch_losses[epoch], epoch_auc[epoch]])
    
    # print best model according to loss and auc
    print(("BEST_MODEL: {} LOSS: {}").format(best_loss_model, best_loss))
    print(("BEST_MODEL: {} AUC: {}").format(best_auc_model, best_auc))
    print("\n")


# Execute training
# OPTIONAL: set your own hyperparameters
train()