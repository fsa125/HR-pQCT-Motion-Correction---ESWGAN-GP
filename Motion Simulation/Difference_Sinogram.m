clc;
clear all;
close all;


Motion_free = imread('samples/Motion_Free/Motion Free Sinogram.jpg');
Motion_corrupted = imread('samples/Motion_Corrupted/Motion Corrupted Sinogram.jpg');

Difference_image = Motion_free - Motion_corrupted;

imshow(imcomplement(2*Difference_image))

