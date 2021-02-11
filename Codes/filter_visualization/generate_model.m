dataDir= './../new_aug_dir/';
Symmetry_Groups = {'Normal', 'Tuberculosis'};

train_folder = 'train_dir';
test_folder  = 'test_dir';

fprintf('\nLoading Train Filenames and Label Data...\n'); t = tic;
train_alex = imageDatastore(fullfile(dataDir,train_folder),'IncludeSubfolders',true,'LabelSource','foldernames');
train_alex.Labels = reordercats(train_alex.Labels,Symmetry_Groups);
fprintf('Done in %.02f seconds\n', toc(t));

fprintf('Loading Test Filenames and Label Data...'); t = tic;
test_alex = imageDatastore(fullfile(dataDir,test_folder),'IncludeSubfolders',true,'LabelSource','foldernames');
test_alex.Labels = reordercats(test_alex.Labels,Symmetry_Groups);
fprintf('Done in %.02f seconds\n', toc(t));


batchSize = 4;
numEpochs = 20;

rng('default');

layers = [
    imageInputLayer([224 224 3]); 
    convolution2dLayer(3,64,'Padding','same');
    batchNormalizationLayer; 
    reluLayer(); 
    convolution2dLayer(3,64,'Padding','same');
    reluLayer();
    batchNormalizationLayer;
    reluLayer(); 
    maxPooling2dLayer(2,'Stride',2);

    convolution2dLayer(3,128,'Padding','same');
    batchNormalizationLayer;
    reluLayer();
    convolution2dLayer(3,128,'Padding','same');
    batchNormalizationLayer;
    reluLayer();
    maxPooling2dLayer(2,'Stride',2);

    convolution2dLayer(3,256,'Padding','same');
    batchNormalizationLayer;
    reluLayer();
    convolution2dLayer(3,256,'Padding','same');
    batchNormalizationLayer;
    reluLayer();
    convolution2dLayer(3,256,'Padding','same');
    batchNormalizationLayer;
    reluLayer();
    maxPooling2dLayer(2,'Stride',2);
 
    convolution2dLayer(3,512,'Padding','same');
    batchNormalizationLayer;
    reluLayer();
    convolution2dLayer(3,512,'Padding','same');
    batchNormalizationLayer;
    reluLayer();
    maxPooling2dLayer(2,'Stride',2);

    fullyConnectedLayer(1024); 
    dropoutLayer(0.3);

    fullyConnectedLayer(2); 
    softmaxLayer(); 
    classificationLayer();	
    ];


numIterationsPerEpoch = floor(numel(train_alex.Labels)/batchSize);
options = trainingOptions('sgdm',...
    'MiniBatchSize',batchSize,...
    'MaxEpochs',numEpochs,...
    'InitialLearnRate',1e-5);

[net1,info1] = trainNetwork(train_alex,layers,options);
save('vgg_mod2.mat','net1');


    
    

