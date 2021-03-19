# Cornell-Birdcall-Identification
https://www.kaggle.com/c/birdsong-recognition  
Complete training pipeline using CNNs with melspectrograms as inputs. Make sure you are running on pytorch 1.6 with automatic mixed precision training supported and pytorch lightning < 1.0.0. Credit to my teammate hktxt.
 
## How to run
1. ```pip install -r requirements.txt```  
2. go to ```data``` fold and run ```resample.py```. Make sure that you have modified the path accordingly.
3. run ```train.py```.

## Models we support
1. efficientnet b0-3(30 epochs takes around 3 hours, 4 hours, 5.5 hours and 7 hours, respectively)
2. resnest 50(30 epochs takes 4 hours)
3. resnet 50
4. se_resnext50_32x4d
5. pyconvhgresnet
6. resnet_sk2

## Loss function choices
1. Topk loss
2. Angle loss

## Augmentations
1. spec augmentation
2. mixup
3. cutmix

## Other techiniques
1. vgg-head: use Conv2d and avg pooling layers instead of global avg pooling
2. sample balancing
3. cosine annealing learning rate with T_max=10.
