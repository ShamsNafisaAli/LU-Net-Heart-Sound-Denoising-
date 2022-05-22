%% Run the HMM on an unseen test recording:
% And display the resulting segmentation
function estim=estimation(data_r,file_name)
% data_r=audioread('44_5.wav');
% data_r=resample(data_r,1000,4000);

numPCGs = 1;
%%
springer_options = default_Springer_HSMM_options;

%% Load the audio data_r and the annotations:
% These are 6 example PCG recordings, downsampled to 1000 Hz, with
% annotations of the R-peak and end-T-wave positions.
load('example_data.mat');

%% Split the data_r into train and test sets:
% Select the first 5 recordings for training and the sixth for testing:
train_recordings = example_data.example_audio_data([1:6]);
train_annotations = example_data.example_annotations([1:6],:);

% test_recordings = example_data.example_audio_data(6);
% test_annotations = example_data.example_annotations(6,:);


%% Train the HMM:
[B_matrix, pi_vector, total_obs_distribution] = trainSpringerSegmentationAlgorithm(train_recordings,train_annotations,springer_options.audio_Fs, false);


%%
for PCGi = 1:numPCGs
    [assigned_states] = runSpringerSegmentationAlgorithm(data_r, springer_options.audio_Fs, B_matrix, pi_vector, total_obs_distribution, false);
% l=[file_name,'.png'];
%     saveas(gcf,l)
%     close(gcf)
end
noise_signal=data_r;%%carefully observed
noise_real_signal=data_r;
%%
noise_signal([find(assigned_states==1)' ,find(assigned_states==3)'])=0;
noise_power=sum(power(noise_signal,2))/length(find(noise_signal));
%%
noise_real_signal([find(assigned_states==2)' ,find(assigned_states==4)'])=0;
noise_real_signal_power=sum(power(noise_real_signal,2))/length(find(noise_real_signal));
signal_power=noise_real_signal_power-noise_power;
estim=10*log10(signal_power/noise_power);

end
