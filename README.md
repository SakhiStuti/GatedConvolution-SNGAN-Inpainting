# GatedConvolution-SNGAN-Inpainting
This is a pytorch implementation of "Free-Form Image Inpainting with Gated Convolution". It performs image inpainting for irregular masks. Contextual Attention has not be added yet. These results are without the attention layer.

## Some Results
![Results](https://github.com/SakhiStuti/GatedConvolution-SNGAN-Inpainting/blob/master/results/24.png)
![Results](https://github.com/SakhiStuti/GatedConvolution-SNGAN-Inpainting/blob/master/results/32.png)
![Results](https://github.com/SakhiStuti/GatedConvolution-SNGAN-Inpainting/blob/master/results/44.png)
![Results](https://github.com/SakhiStuti/GatedConvolution-SNGAN-Inpainting/blob/master/results/48.png)  
Images from left to right - input image, course network output, fine network output, completed image, groundtruth

## Training from scratch
Train and Validation image roots can be changed in the train_1.yml file. Then you can train the network using the command-  
python train.py

## Testing a folder of images with random masks
Run python test_folder.py --image_root X --output_root Y --checkpoint_path Z  

The result images will be saved in output_root

## Testing an image mask pair
Run python test_file.py --image_path X --mask_path Y --output_path Z --checkpoint_path W

## References
* https://github.com/JiahuiYu/generative_inpainting
* https://github.com/avalonstrel/GatedConvolution_pytorch
