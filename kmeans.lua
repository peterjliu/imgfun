require 'image';
require 'imgfun';

o = image.load('obama.jpg'); 
newimg = imgfun.gen_kmeans_image(o, 3)
image.save('obama-3.jpg', newimg)
