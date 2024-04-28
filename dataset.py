import numpy as np
import pandas
import torch 
from torch.utils.data import Dataset
import SimpleITK as sitk
import cv2

# the dataset classes use package SimpleITK:
# R. Beare, B. C. Lowekamp, Z. Yaniv, “Image Segmentation, Registration and Characterization in R with SimpleITK”, J Stat Softw, 86(8), doi: 10.18637/jss.v086.i08, 2018.
# Z. Yaniv, B. C. Lowekamp, H. J. Johnson, R. Beare, “SimpleITK Image-Analysis Notebooks: a Collaborative Environment for Education and Reproducible Research”, J Digit Imaging., doi: 10.1007/s10278-017-0037-8, 31(3): 290-303, 2018.
# B. C. Lowekamp, D. T. Chen, L. Ibáñez, D. Blezek, “The Design of SimpleITK”, Front. Neuroinform., 7:45. doi: 10.3389/fninf.2013.00045, 2013.

# MIP dataset returns 1-channel tensor containing MIP image of patient PET scan
# along the specified dimension(0/1 - coronal or sagittal), and label (0/1 - whether the patient has malignant lymphoma)
# The csv file is expected to have fields in such order:
# pid (patient id), pet(path to PET.nii.gz), ct(path to CT.nii.gz), mask(path to MASK.nii.gz), class (0/1)
class MIP(Dataset):
    def __init__(self, csv_file="./data/train_data.csv", dim=1, transform=None):
        self.csv_file = pandas.read_csv(csv_file)
        self.transform = transform
        self.dim = dim

    def __len__(self):
        return len(self.csv_file)

    def __getitem__(self, idx):
        pet = self.csv_file.iloc[idx, 1]
        label = self.csv_file.iloc[idx, 4]

        label = torch.tensor([label], dtype=torch.float32)

        out = preprocess_pet(pet=pet,dim=self.dim)
        if self.transform is not None:
            out = self.transform(out)

        return out, label

# MIP_MNN is only used for testing the multi model, and is not designed for training purposes
class MIP_MNN(Dataset):
    def __init__(self, csv_file="./data/test_data.csv"):
        self.csv_file = pandas.read_csv(csv_file)
    
    def __len__(self):
        return len(self.csv_file)
    
    def __getitem__(self, idx):
        label = self.csv_file.iloc[idx, 4]
        label = torch.tensor([label], dtype=torch.float32)

        pet = self.csv_file.iloc[idx, 1]
        out0 = preprocess_pet(pet, dim=0)
        out1 = preprocess_pet(pet, dim=1)
        return out0, out1, label

# only used for the 1. round of RIS competition, returns patient id and 
# MIP images along dimensions 0 nd 1
# the csv file is expected to have fields:
# pid (patient id), pet (path to PET.nii.gz), ct (path to CT.nii.gz)
class RIS_1(Dataset):
    def __init__(self, csv_file):
        self.csv_file = pandas.read_csv(csv_file)
    
    def __len__(self):
        return len(self.csv_file)

    def __getitem__(self, idx):
        patient = self.csv_file.iloc[idx, 0]
        pet = self.csv_file.iloc[idx, 1]
        out0 = preprocess_pet(pet, dim=0)
        out1 = preprocess_pet(pet, dim=1)
        return out0, out1, patient

# preprocesses PET scan into desired format
def preprocess_pet(pet, dim):
    mip = make_mip(image_path=pet, dim=dim)
    out = resize_mip(mip=mip)
    out = normalize(img=out, min_v=0, max_v=60)
    out = torch.from_numpy(out)
    return out

# resize MIP images to desired (square) size
def resize_mip(mip, size=256):
    out = np.zeros((1, size, size), dtype="float32")
    mip = square_2d(mip)
    mip = cv2.resize(mip, dsize=(size, size), interpolation=cv2.INTER_LINEAR)
    out[0] = mip
    return out

# creates coronal and sagittal MIP images from nii.gz
def make_mip(image_path, dim):
    image = sitk.ReadImage(image_path)
    image_size = image.GetSize()

    projection = sitk.MaximumProjection(image, dim)
    p = sitk.GetArrayFromImage(projection)
    if dim == 0:
        p = p[:,:,0]
    if dim == 1:
        p = p[:,0,:]

    return p

# fills the image along the shorter dimension to square
def square_2d(img):
    y = img.shape[0]
    x = img.shape[1]

    if y < x:
        dif = int((x - y)/2)
        new_img = np.zeros((y + dif*2, x)).astype("float32")
        new_img[dif:y + dif, :] = img
        return new_img
    elif x < y:
        dif = int((y - x)/2)
        new_img = np.zeros((y, x + dif*2)).astype("float32")
        new_img[:, dif:x + dif] = img
        return new_img
    else:
        return img

# normalizes the image based on the desired min and max values
def normalize(img, min_v, max_v):
    img[img < min_v] = min_v
    img[img > max_v] = max_v
    img = img.astype("float32")
    img = (img - min_v)/(max_v - min_v)

    return img 
