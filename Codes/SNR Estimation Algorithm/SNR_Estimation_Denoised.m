clc
clear all
close all

T1= readtable('F:\NAFISA DOCS\BUET Academics\L4 T1 study things\BME 400\Data\Pascal\Pascal_Enhanced\pascal_unet.csv');
Fs = 1000;
tk1=T1(2:end,2:end);

ar1=table2array(tk1);
[M, N] = size(ar1);
for i=1:M
    y1=ar1(i,:);
    y1 = y1';
    y1 = y1/max(y1);
    k(i)=estimation(y1);
end

disp('The mean of the predicted SNR:')
mean(k)
disp('The std of the predicted SNR:')
std(k)
