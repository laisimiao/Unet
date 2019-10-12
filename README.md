# Unet
参考博客：https://blog.csdn.net/awyyauqpmy/article/details/79290710  
在不数据增强的情况下已经能够训练了，并且取得了85%左右的训练/验证准确度 
因为没太搞懂数据增强的那部分（跑是可以跑，但是没实现很好的一个增强效果）--2019/10/12
这里放一张训练结果图： 
![unet.py训练结果图](https://ae01.alicdn.com/kf/Hab913980059d4a53829a43f31a893d205.png) 
# My env
1. ubuntu18.04 + GTX1080 
2. keras2.2.5 + tensorflow-gpu1.13.1 + cuda10.0 + cudnn7.4.2 
# Note 
不要使用windows完成，因为里面有个glob.glob会返回Windows的文件分隔符'\'，这个会导致一个地方出错
