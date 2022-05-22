path_Heart_Train = "./data_set/physionet_Langley+Spectral_Audio/train"
path_Lung_Train = "./data_set/lung_sound_dataset/train"

pathheartVal ='./data_set/Git_hub_dataset/Git_val'
pathlungval ='./data_set/lung_sound_dataset/val'
pathhospitalval ='./data_set/Hospital_noise_filtered_resampled/train'

window_size = 0.8
name_model = "lunet"
input_shape = 800
output_shape = 800
sampling_rate_new = 1000
check = rf"./check_points/{name_model}.h5"
#check=rf"./check_points/{name_model}_model_pc07_PhysioNet_{sampling_rate_new}_-3dB_to_6dB.h5"
