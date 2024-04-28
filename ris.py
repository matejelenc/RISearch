# import packages
from resnet import MNN
import torch
import csv
import os
from dataset import RIS_1
from torch.utils.data import DataLoader
import sys

# creates csv_file which is then passed as argument to the RIS_1 object
def create_ris_data_csv(path="./RIS_test_no_masks"):
    patients = os.listdir(path)
    patients.sort()
    csv_file = "./ris_1_round.csv"
    with open(csv_file, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["pid", "pet", "ct"])
        for patient in patients:
            patient_path = os.path.join(path, patient)
            pet = os.path.join(patient_path, "PET.nii.gz")
            ct = os.path.join(patient_path, "CT.nii.gz")
            writer.writerow([patient, pet, ct])
    
    return csv_file

# generates results for the 1. round of RIS
def create_ris_results_csv(csv_file, model, device):
    dataset = RIS_1(csv_file=csv_file)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    result_file = "./napovedi-RISearch.csv"
    with open(result_file, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["patient", "result"])
        for x0, x1, pid in dataloader:
            x0 = x0.to(device)
            x1 = x1.to(device)
            out = model(x0, x1)
            out = torch.sigmoid(out)
            writer.writerow([pid[0], out.item()])

    return result_file
        
def main():

    # get path to test data
    path = "x" if len(sys.argv) == 1 else sys.argv[1]
    
    # create csv file for easier data loading
    csv_file = create_ris_data_csv(path) if path != "x" else create_ris_data_csv()

    # create device
    device = torch.device("cuda:0" if torch.cuda.is_available else "cpu")

    # load model
    model = MNN()
    model.load_state_dict(torch.load("./ris_models/classifier.pth"))
    model.eval()
    model.to(device)

    # generate results
    result_file = create_ris_results_csv(csv_file, model, device)
    print(("Successfully generated results for 1.round of RIS, which are available at:\n {}").format(result_file))

if __name__ == "__main__":
    main()
