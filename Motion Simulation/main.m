clc;
clear all;
close all;
%Run this code for passing the Source GT images Ys
%it returns the degraded source image, Xs and saves the images in a
%specific directory mentioned by output_directory

input_directory = 'samples/prox rad 1 Test img/'; % Motion grade 1 image

output_directory = 'samples/prox rad Test img sim 2/'; %

readImagesInFolder(input_directory,output_directory);




