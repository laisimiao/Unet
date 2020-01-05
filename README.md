# Unet
In this branch, it's written in Keras and mostly copy from: https://github.com/decouples/Unet  
Because my research direction is not Biomedical Image Segmentation, I do this just for interest and I repaire a mis-match problem between test and predict images. So, I re-produce it in Pytorch 1.0.0([pytorch branch](https://github.com/laisimiao/Unet/tree/pytorch)), and use ISBI dataset(http://brainiac2.mit.edu/isbi_challenge/). You can also download from my link below in **How to run**.  
**ps**: I don't use any data augmentation methods. My chinese blog is here. 
# My env
1. ubuntu18.04 + GTX TITAN XP  
2. keras2.2.4 + tensorflow-gpu1.13.1 + cuda10.0 + cudnn7.4.2  
3. torch1.0.0 + torchvision0.2.0 

# How to run 
In this branch, you first need to create a new folder in this directory named **results** for saving model predicted pictures, like in Linux, you can run command in terminal:  
```mkdir results```  
Next step is: you need download ISBI dataset([baidu drive](https://pan.baidu.com/s/10jsOj0XXc3A6RqdkT8VYDQ) pswd:jwix or [google drive](https://drive.google.com/open?id=1c20QNqo5earWk4HKe_VGwFdBpY959Mwb)) and extract it in this directory named **raw** in which there has three sudfolders: train, label and test.   
Then you can run the code.
# Results display
Now I display a 7-th results from pytorch branch:  
![6-pytorch.png](https://i.loli.net/2020/01/05/VPOWDj5Qzs1NIlo.png)  

# Note
In windows, it has some small mistakes with **glob.glob**, but in pytorch branch it has no this bug.
