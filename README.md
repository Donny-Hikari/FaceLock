# FaceLock  
Recognize your face, lock your computer when you are absent or when others appear.  

## Go through  
When you leave your computer, FaceLock lost track of you.  
When somebody else come into the vesion of your computer, and last for sometime, FaceLock will begin to count down.  
![countdown](./readme_res/main_detected.png)  
When count down to MaxFailDelay, FaceLock lock your computer automatically.  
![lockup](./readme_res/main_lockup.png)  

## Accuracy  
The valid result is as follow. Please be aware that it's just tested with limit cases and may be different vary to images inputted.  
Valid result using Adam optimizer:  
![valid_result_adam](./readme_res/valid_result_adam.png)  
Valid result using SGD optimizer:  
![valid_result](./readme_res/valid_result.png)  
From the valid result we can see that Adam reaches better accuracy in less epochs comparing to SGD.  
  
So far, FaceLock has never mistake others for me during my test after I add some layers and increase epoch. Although sometimes it might fail to recognize me if I make faces.  
Please let me know if in your case FaceLock happens to mistake someone else for you.  

## Requirements  
* Python3.5  
* Windows 10  
* WebCamera  
* Lots of images of you and others  

Put the training images to [data/train/me](./data/train/me) and [data/train/other](./data/train/other).  
Put the validation images to [data/valid/me](./data/valid/me) and [data/valid/other](./data/valid/other).  
Please notice that validating image should be different from training image. Validating serves as a simulation for real time input.  
In case you are puzzled about how much images will be enought, I am using about 160 images of myself and 290 images of others.  

## Preparation  
Install Tensorflow, Keras, OpenCV.  
```  
> pip install tensorflow  
> pip install keras  
> pip install opencv-python  
```  
Remember to use tensorflow backend for keras (instead of Theano).  

## Usage  
First of all, train the network.  
```  
> python train.py  
```  
  
Then, validate the network.  
```  
> python valid.py  
```  
  
Finally, start FaceLock.  
```  
> python main.py  
```  

To start FaceLock in background, use pythonw on windows.  
```  
> pythonw \path\to\FaceLock\main.py 1>stdout.txt 2>stderr.txt
```  

# About Gray Mode  
Comparing to [Hironsan](https://github.com/Hironsan)'s project [BossSensor](https://github.com/Hironsan/BossSensor), I use grayscale images instead of BGR images. In my opinion, neutral network trained with grayscale images has better fitness, and is less affected by the environment. Of course, it takes longer to train. So I reduce the dropout rate and increase the training epoch. It turns out to work fine.  
You can switch between gray mode and BGR mode, simply change GRAY_MODE in input.py to True or False.  
``` python  
# input.py  
GRAY_MODE = True  
```  

## Special Thanks  
Thank to the inspiration from [Hironsan](https://github.com/Hironsan), whose project [BossSensor](https://github.com/Hironsan/BossSensor) inspires me a lot, and lead to the birth of my FaceLock project.  

## Licence  
[MIT Licence](./LICENSE)  

## Author  
[Donny-Hikari](https://github.com/Donny-Hikari)  