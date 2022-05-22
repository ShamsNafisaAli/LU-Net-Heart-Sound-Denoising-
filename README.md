#  Robust Deep Learning Framework for Real-Time Denoising of Heart Sound

## Requirements
- Python 
- Matlab 
- Keras 
- Tensorflow 
- Sklearn 
- Tensorboard
- naturalsort
- keras-flops
- librosa
- soundfile
- numpy
- matplotlib

## How To Run
### Data Preparation:
First download the data folder from this GoogleDrive Link
Place Physionet dataset (not included in the provided data folder) in the corresponding folders inside the data/physionet/training folder. The csv files containing the labels should be put inside the corresponding folders inside the labels folder and all of them should have the same name, currently 'REFERENCE_withSQI.csv'. If you change the name you'll have to rename the variable labelpath in extract_segments.m and extract_segments_noFIR.m
Run extract_segments_noFIR.m it first then run data_fold_noFIR.m to create data fold in mat format which will be loaded by the model for training and testing. fold0_noFIR.mat is given inside data/feature/folds for convenience, so that you don't have to download the whole physionet dataset and extract data for training and testing.

### Training:
For Training run the trainer.py and provide a dataset name (or fold name) i.e. fold0_noFIR. The command should be like this :

python trainer.py fold0_noFIR
Other parameters like epochs, verbose, batch_size, pre-trained model path can be passed as arguments.

python trainer.py fold0_noFIR --epochs 300 --batch_size 1000 

### Re-Generate Results:
Run the heartnet testbench.ipynb on Jupyter Notebook from the beginning until the block named Model.Predict . Select a log_name by uncommenting one from the LOG name block. The trained models for "heartnet type2 tconv" and "potes algorithm" is given in the log and model directory. These models are trained on fold0_noFIR which is included in the data folder.
To do the McNemer test read the instruction given in the LOG name block of the notebook. To plot roc curve run the ROC curve block.
