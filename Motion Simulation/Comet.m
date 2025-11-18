function smearedImage = Comet (input_image,theta,kernel)
%%https://stackoverflow.com/questions/57326712/how-to-blur-an-image-in-one-specific-direction-in-matlab
L = kernel; % kernel width
sx=3;
sy=10;


I = input_image;
x = -L:1.0:L;

[X,Y] = meshgrid(x,x);
rX = X.*cos(theta)-Y.*sin(theta);
rY = X.*sin(theta)+Y.*cos(theta);
H1 = exp(-((rX./sx).^2)-((rY./sy).^2));
Hflag = double((0.*rX+rY)>0);
H1 = H1.*Hflag;
comet_kernel = H1/sum((H1(:)));

smearedImage = conv2(double(I),comet_kernel,'same');

%imshow(smearedImage,[]);
%imshow(smearedImage,[]);



end