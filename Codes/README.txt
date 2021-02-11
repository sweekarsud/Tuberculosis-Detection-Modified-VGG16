NOTE 1: Data is not attached in this zip-folder, kindly download it from the OneDrive link given below:
https://pennstateoffice365-my.sharepoint.com/:f:/g/personal/sks6492_psu_edu/EmHqvByTWs1IpF9QLU3gjR0BrSk1RCtO4FYeq6UkbMe9Eg?e=zUHdpq 

After downloading the data folder of size 4 GB, kindly place it in the Codes/ folder without changing 
the name or any convention as the codes are accustomed to the format of the data directly without any 
modifications.

NOTE 2: In order to run the codes present in the folder kindly ensure that the dependency list version is 
satisfied mentioned in the Dependency_List file.

NOTE 3: Kindly run the codes in the order presented here because the output of one code is dependent on
another. In addition, no need to change the directory structure for running any code as the paths are 
well managed in the code itself.

This folder Codes/ consists of the following files:
1) preprocess_data.py : This python file corresponds to the pre-processing code employed in this work.
This code removes black border, resizes, centering and whitening of input images. The output of this code
is proc_data/ folder using input data/ folder. 

2) data_augmentation.py : This python file corresponds to the data-augmentation code employed in this work.
It takes the images generated after pre-processing in proc_data/ and generates the output folder new_aug_dir/
for both training and testing.

3) baseline_vgg16.py : This python file corresponds to the baseline model employed in this work with 
VGG16 pre-trained weights.

4) baseline_resnet50.py : This python file corresponds to the baseline model employed in this work with 
ResNet50 pre-trained weights.

5) ladder_network_code.py : This python file corresponds to the ladder network employed in this work which is
the first proposed model. This code makes use of another code ladder_net.py which contains several function 
necessary for the operation.

5) ladder_net.py : This python file corresponds to the ladder network and this need not be run as it will be 
directly used by ladder_network_code.py

6) modified_VGG16.py : This python file corresponds to the new modified VGG16 network (second proposed model).
This code saves the model weights automatically, which can be further used for visualization.

7) tsne_visualization.py : This python file corresponds to the t-SNE visualization plots generation. The file 
takes in the saved model weights along with the augmented data and generates the t-SNE plots.

8) saliency_map.py : This python file corresponds to the saliency map/heat map plots generation. The file takes 
in the saved model weights along with one chest x-ray sample and generates the heat map.

9) roc_curve_plot.py : This python file corresponds to the ROC and Precision-Recall plots. The file takes in the  
saved model weights along with the augmented data and generates the ROC and Precision-Recall plots.

10) filter_visualization/generate_model.m : This file generates the model in matlab format with the same architecture
as the new modified VGG16 and saves it into .mat format for filter visualization. For running this code kindly use
MATLAB/R2019a.

11) filter_visualization/visualization.m : This file generates the filter visualization for each of the layers and 
saves it. For running this code kindly use MATLAB/R2019a.

