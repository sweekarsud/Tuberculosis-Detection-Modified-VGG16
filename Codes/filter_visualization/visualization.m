load('vgg_mod2.mat');

conv_chann = net.Layers(2,1).NumFilters;
Vis = deepDreamImage(net1,2,1:conv_chann,'PyramidLevels',1);

figure;
montage(Vis);
title('Convolutional Filter 1');

conv_chann = net.Layers(5,1).NumFilters;
Vis = deepDreamImage(net1,5,1:conv_chann,'PyramidLevels',1);

figure;
montage(Vis);
title('Convolutional Filter 2');

conv_chann = net.Layers(10,1).NumFilters;
Vis = deepDreamImage(net1,10,1:conv_chann,'PyramidLevels',1);

figure;
montage(Vis);
title('Convolutional Filter 3');

conv_chann = net.Layers(13,1).NumFilters;
Vis = deepDreamImage(net1,13,1:conv_chann,'PyramidLevels',1);

figure;
montage(Vis);
title('Convolutional Filter 4');

conv_chann = net.Layers(17,1).NumFilters;
Vis = deepDreamImage(net1,17,1:conv_chann,'PyramidLevels',1);

figure;
montage(Vis);
title('Convolutional Filter 5');

conv_chann = net.Layers(20,1).NumFilters;
Vis = deepDreamImage(net1,20,1:conv_chann,'PyramidLevels',1);

figure;
montage(Vis);
title('Convolutional Filter 6');

conv_chann = net.Layers(23,1).NumFilters;
Vis = deepDreamImage(net1,23,1:conv_chann,'PyramidLevels',1);

figure;
montage(Vis);
title('Convolutional Filter 7');

conv_chann = net.Layers(27,1).NumFilters;
Vis = deepDreamImage(net1,27,1:conv_chann,'PyramidLevels',1);

figure;
montage(Vis);
title('Convolutional Filter 8');

conv_chann = net.Layers(30,1).NumFilters;
Vis = deepDreamImage(net1,30,1:conv_chann,'PyramidLevels',1);

figure;
montage(Vis);
title('Convolutional Filter 9');

