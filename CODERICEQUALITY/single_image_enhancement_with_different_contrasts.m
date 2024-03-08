a = imread('img2.jpg');


select_low  = 0.0;
select_high = 0.5;

output_low  = 0.0;
output_high = 1.0;

for im = 1:6;
    low_in = select_low + double(im/13);
    name = sprintf('images/ut/out_%f.jpg', low_in);
    
    
    n = imadjust(a, [low_in select_high], [output_low output_high]);
    imwrite(n, name) 
end;
    


