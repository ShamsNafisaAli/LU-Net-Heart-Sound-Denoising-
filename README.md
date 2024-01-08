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
- pandas
- matplotlib

***

## How To Run
<!-- ### Data Preparation:
- First download the PHS Data (Processed) and ICBHI Dataset (Processed) folder from this GoogleDrive Link
- Update the definition of path_Heart_Train and path_Lung_Train of config.py inside Codes folder
Place Physionet dataset (not included in the provided data folder) in the corresponding folders inside the data/physionet/training folder. The csv files containing the labels should be put inside the corresponding folders inside the labels folder and all of them should have the same name, currently 'REFERENCE_withSQI.csv'. If you change the name you'll have to rename the variable labelpath in extract_segments.m and extract_segments_noFIR.m
Run extract_segments_noFIR.m it first then run data_fold_noFIR.m to create data fold in mat format which will be loaded by the model for training and testing. fold0_noFIR.mat is given inside data/feature/folds for convenience, so that you don't have to download the whole physionet dataset and extract data for training and testing. -->

***

### Training:
- First download the PHS Data (Processed) and ICBHI Dataset (Processed) folder from GoogleDrive Link provide inside Data/data_download_link.txt file
- Update the definition of path_Heart_Train and path_Lung_Train and specify the model name (for example: use 'lunet' for proposed denoising framework) under the Codes/config.py file
- Run Codes/train_model.py to start the training

***

### Re-Generate Results:
#### Open-access Heart Sound Dataset
- First download the OAHS Dataset, Hospital Ambient Noise (HAN) Dataset, and ICBHI Dataset (Processed) folders from GoogleDrive Link provide inside Data/data_download_link.txt file 
- Update the definition of pathheartVal, pathlungval and pathhospitalval under the Codes/config.py file
- Put the directory of training weight(you can find pretrained weight inside Models folder) inside Codes/result_making.py file 
- Run Codes/result_making.py to start the inference


#### PASCAL Heart Sound Challenge Dataset
- First download the PaHS Dataset provided inside Data/data_download_link.txt file
- Update the definition of pathheartVal, pathlungval and pathhospitalval under the Codes/config.py file
- Put the directory of training weight (you can find pretrained weight inside Models folder) inside Codes/result_making.py file 
- Run Codes/result_making.py to start the inference
- Use the directory of the generated .csv file (containing the denoised audio samples) inside the readtable function of Run Codes/SNR Estimation Algorithm/SNR_Estimation_Denoised.m to get the estimated SNRs for the denoised signals
- Run Codes/SNR Estimation Algorithm/SNR_Estimation_Noisy.m to get the estimated SNRs for the noisy signals

***

## Citation

If this repository helped your research, please cite:<br />

Ali, S. N., Shuvo, S. B., Al-Manzo, M. I. S., Hasan, A., & Hasan, T. (2023). An end-to-end deep learning framework for real-time denoising of heart sounds for cardiac disease detection in unseen noise. IEEE Access.

<blockquote>
  
@article{ali2023end, <br />
  title={An end-to-end deep learning framework for real-time denoising of heart sounds for cardiac disease detection in unseen noise},<br />
  author={Ali, Shams Nafisa and Shuvo, Samiul Based and Al-Manzo, Md Ishtiaque Sayeed and Hasan, Anwarul and Hasan, Taufiq},<br />
  journal={IEEE Access},<br />
  year={2023},<br />
  publisher={IEEE}<br />
}
