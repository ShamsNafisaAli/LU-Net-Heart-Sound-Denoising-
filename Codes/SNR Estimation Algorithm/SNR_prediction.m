clc
clear all
close all

signal_path = dir(['F:\NAFISA DOCS\BUET Academics\L4 T1 study things\BME 400\Data\HS_Quality Assessment\heart_sound_database_for_quality\Dataset Wise Division\PhysioNet CinC Full\5\*.wav']);
nfiles = length(signal_path);
Fs = 1000;
%%
k=zeros(nfiles,1);
for i=1: nfiles
    currentfilename1=signal_path(i).name;
    f1 = fullfile('F:\NAFISA DOCS\BUET Academics\L4 T1 study things\BME 400\Data\HS_Quality Assessment\heart_sound_database_for_quality\Dataset Wise Division\PhysioNet CinC Full\5\',currentfilename1);
    [y1,F1] = audioread(f1);
    y1= resample(y1,Fs,F1);
    k(i) = estimation(y1,currentfilename1);
end
