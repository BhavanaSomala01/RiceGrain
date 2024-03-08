f = imread('img2.jpg');
figure,imshow(f)

%convert the image to black and white 
%image i.e a binary image

bw = im2bw(f);
figure,imshow(bw)
%Using canny edge detector, detect the edges of the 
%rice grains.
%use threashold matrix T, to adjust the threashholsd and sigma for canny
%edge detector.
%
%We emperically agreed on the follwoing values

T       = [0.1 0.5];
sigma   = 1; 


imedge = edge(bw, 'canny', T, sigma);


% Using imfill fill all the regions
% to get closed objects 

imf = imfill(imedge, 'holes');


% Label all the objects which are detected by BWlabel
% g is a labled image, and n is number of objects

[g n] = bwlabel(imf);



% using regionprops, get the properties of each object
% calculate the desired properties
% from region props

stats = regionprops(g);


stats = regionprops(bw,'Centroid', 'MajorAxisLength','MinorAxisLength')


averageHeight  = 0;
averageWidth = 0;
for i = 1:numel(stats);    
   averageHeight = averageHeight + stats(i).MajorAxisLength;
   averageWidth = averageWidth + stats(i).MinorAxisLength;   
end;

averageWidth = averageWidth/numel(stats);
averageHeight = averageHeight/numel(stats);



% collect the details for each detected rice grain from regionprops in
% tabular format
tabular = regionprops('table', bw,'Centroid', 'MajorAxisLength','MinorAxisLength');






display('Summary');
display(tabular);
display(sprintf('Number of objects in the image are %d', n));
display(sprintf('Average Width of each grain is Width = %f', averageWidth));
display(sprintf('Average Height of each grain is Height = %f', averageHeight));










