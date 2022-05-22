clc
clear all
close all
%%
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
%%
% disp('The mean of the predicted SNR:')
% mean(k)
% disp('The std of the predicted SNR:')
% std(k)
%%
label_x = [1, 2, 3, 4, 5];
snr_y = [0.9875, 3.3463, 6.9256, 11.3858, 15.2999];
mdl = fitlm(label_x,snr_y)
tbl = anova(mdl,'summary')

%%
y = audioread('50_1.wav');
z = y(10001:20000);
plot(z)
xlim([0 20000])
ylim([-1 1])
xlabel("No. of Samples", fontsize=14)
ylabel("Amplitude", fontsize=14)
exportgraphics(gca,'50_2.jpg','BackgroundColor','white', 'Resolution',600)

