# Introduction
This is the code that was used to train the models and generate results for the 1. round of RIS competition. The goal of the 1. round was to classify patients as class 0 (healthy) or class 1 (has malignant lymphoma). Our model achieved an AUC score of 0.98 which can be computed from the `.csv` files `./resitve_krog1_24.csv` (the actual results) and `./napovedi-RISearch.csv` (the predictions).

If someone were to use this code to train and test their own models, it should be noted that the files in `./data` contain the paths to patients' files relative to the computer that was used for training and on which the patients' files were located, therefore one should regenerate this files according to their own dataset and where it is stored on their device.

 # IMPORTANT 
 Due to file size limitations of Github, the actual models have not been committed and are available at [ris models](https://drive.google.com/drive/folders/1NajVEMzzBgibIXG42ffQhVn6cKvEz6LX?usp=drive_link).
 
# Requirements
It is expected of the user to have already installed Python and pip. To download all necessary python packages run: `./requirements.sh`

# Generating Results
Since the dataset for the 1. round of RIS has already been uploaded to Sling, it is not present in this folder.
To generate results for the 1. round of the RIS competition, run `python ris.py <path_to_dataset>`. The default path in the `ris.py` file to the **RIS_test_no_masks** dataset is set to  `./RIS_test_no_masks`, therefore if the dataset is present in this folder, the results can also be generated using only `python ris.py`.

# Data Augmentation
Since the dataset available was relatively small and unbalanced, additional data was acquired with the [TCIA dataset](https://www.cancerimagingarchive.net/). The data was then split into about 450 patients for training and 80 for testing. Both splits had 1:1 ratio of patients with and without malignant tumours. Additionally, to improve the training of the model, random horizontal flips and rotations to up to 30° were applied to the data.

# Data Preprocessing
In training the models, only PET scans were used. To improve the efficiency of the training and minimize the complexity of the model, 2 MIP (maximum intensity projection) images were generated for each PET scan, one being the coronal MIP and one sagittal MIP and then resized to size (1, 256, 256). 

# Model Structure
The base model architecture used was ResNet34. Firstly, two separate ResNet34 models were trained, one on coronal MIP images and one on sagittal MIP images (available at `./ris_models` as `coronal.pth` and `sagittal.pth`). To generate more stable results, a MNN (multi model) model was created, which ouputs the average of the sagittal and the coronal model (avaliable at `./ris_models/classifier.pth`). **IMPORTANT**: Due to file size limitations of Github, the actual models have not been committed and are available at [ris models](https://drive.google.com/drive/folders/1NajVEMzzBgibIXG42ffQhVn6cKvEz6LX?usp=drive_link).

# Hyperparameters
Due to time constraints, the effects of combinations of different hyperparameters on the training were not explored very deeply. The final models were trained with batch size of 4, learning rate of 1e-5 and total number of epochs being around 60. The final model was then chosen based upon the best epoch, since its performance on test data was evaluated after each epoch.

# Folder Structure
**Folders**
1. `./data` folder contains `.csv` files that were used for training and testing (`train_data.csv`and `test_data.csv` respectively). All files have the same fields: [pid (patient id), pet (path to PET.nii.gz), ct (path to CT.nii.gz), mask(path to MASK.nii.gz), class (0/1)].
2. `./training_models` and `./training_results` were used during training to save the model state during certain epochs and to save the training results, for further analysis after training and are currently empty.
3. `./roc_curves` contains `.png` images of ROC curves of the final models, that was generated with the test data split (the final models had an AUC of about 0.998 when tested on the test data split).
4. `./ris_models` contains the final models, `coronal.pth` and `sagittal.pth` being of type resnet.ResNet and `classifier.pth` being of type resnet.MNN, which is also the one which was used to generate results for the 1. round of the competition.

**Bash**
1. `train.sbatch` was used to send jobs to Sling.
2. `requirements.sh` can be used to download all necessary python packages.

**Python**
1. `dataset.py` contains dataset classes that inherit the `torch.utils.data.Dataset` class, which were used for training and testing.
2. `train.py` was used for training.
3. `resnet.py` contains all model architecture.
4. `evaluate.py` contains evaluation functions which where used to determine the performance of models.
5. `ris.py` was used to generate results for the 1. round of RIS competition.



# Citations
**SimpleITK** 

1. R. Beare, B. C. Lowekamp, Z. Yaniv, “Image Segmentation, Registration and Characterization in R with SimpleITK”, J Stat Softw, 86(8), doi: 10.18637/jss.v086.i08, 2018.
2. Z. Yaniv, B. C. Lowekamp, H. J. Johnson, R. Beare, “SimpleITK Image-Analysis Notebooks: a Collaborative Environment for Education and Reproducible Research”, J Digit Imaging., doi: 10.1007/s10278-017-0037-8, 31(3): 290-303, 2018.
3. B. C. Lowekamp, D. T. Chen, L. Ibáñez, D. Blezek, “The Design of SimpleITK”, Front. Neuroinform., 7:45. doi: 10.3389/fninf.2013.00045, 2013.  

**scikit-learn** 

4. Scikit-learn: Machine Learning in Python, Pedregosa et al., JMLR 12, pp. 2825-2830, 2011.  

**ResNet34** 

5. He, K., Zhang, X., Ren, S., & Sun, J. (2015). Deep Residual Learning for Image Recognition. arXiv preprint arXiv:1512.03385. Retrieved from https://arxiv.org/pdf/1512.03385.pdf  

**use of MIP-s** 

6. Häggström I, Leithner D, Alvén J, Campanella G, Abusamra M, Zhang H, Chhabra S, Beer L, Haug A, Salles G, Raderer M, Staber PB, Becker A, Hricak H, Fuchs TJ, Schöder H, Mayerhoefer ME. Deep learning for [18F]fluorodeoxyglucose-PET-CT classification in patients with lymphoma: a dual-centre retrospective analysis. Lancet Digit Health. 2024 Feb;6(2):e114-e125. doi: 10.1016/S2589-7500(23)00203-0. Epub 2023 Dec 21. PMID: 38135556; PMCID: PMC10972536.

# Help
For any questions, problems, please refer to: **mj66414@student.uni-lj.si**.
