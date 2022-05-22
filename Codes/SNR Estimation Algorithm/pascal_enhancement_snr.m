clc
clear all
close all
%
signal_path = dir(['F:\NAFISA DOCS\BUET Academics\BSc\L4 T1 study things\BME 400\Data\HS_Quality Assessment\heart_sound_database_for_quality\Dataset Wise Division\PhysioNet CinC Full\1\*.wav']);
nfiles1 = length(signal_path);
Fs = 1000;
%%
k=zeros(nfiles1,1);
for i=1:nfiles1
    currentfilename1=signal_path(i).name;
    f1 = fullfile('F:\NAFISA DOCS\BUET Academics\BSc\L4 T1 study things\BME 400\Data\HS_Quality Assessment\heart_sound_database_for_quality\Dataset Wise Division\PhysioNet CinC Full\1\',currentfilename1);
    [y1,F1] = audioread(f1);
    y1= resample(y1,Fs,F1);
    k(i)=estimation(y1,currentfilename1);
end
%%
% T1= readtable('F:\NAFISA DOCS\BUET Academics\L4 T1 study things\BME 400\Data\Pascal\Pascal_Enhanced\pascal_unet.csv');
% Fs = 1000;
% tk1=T1(2:end,2:end);
% 
% ar1=table2array(tk1);
% [M, N] = size(ar1);
% for i=1:M
%     %     currentfilename1=signal_path(i).name;
%     %     f1 = fullfile('D:\all_upwork_file\thesis_400\archive (4)\Processed_6\Processed_6\raw_2400_val\',currentfilename1);
%     %     [y1,F1] = audioread(f1);
%     %     y1= resample(y1,Fs,F1);
%     y1=ar1(i,:);
%     y1 = y1';
%     y1 = y1/max(y1);
%     k(i)=estimation(y1);
% end

disp('The mean of the predicted SNR:')
mean(k)
disp('The std of the predicted SNR:')
std(k)


