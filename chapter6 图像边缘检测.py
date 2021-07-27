################ 图像边缘检测 #################
#边缘检测一般是在灰度图的基础上进行的
import cv2
import matplotlib.pyplot as plt #绘图展示的一个包
import numpy as np

img = cv2.imread('Resources/lena.jpg',cv2.IMREAD_GRAYSCALE) #读取一张图片 并进行了灰度变换

v1=cv2.Canny(img,80,150) #80 minVal越小 拿到的边界希望就越大，信息相对来说就更少 150 maxVal越大 拿到的边界点就越多 信息就相对更少
v2=cv2.Canny(img,50,100)

res = np.hstack((v1,v2))
cv2.imshow('res image',res)

cv2.imshow('image',img) #两个参数 第一个参数是显示图像的名字 第二个参数是读取的图像

cv2.waitKey(0) #0代表图像一直显示 1000表示是1000ms 延时1s进行自动关闭
cv2.destroyAllWindows() #触发一些按键可以执行关闭



################ 图像边缘检测 #################
#边缘检测一般是在灰度图的基础上进行的
import cv2
import matplotlib.pyplot as plt #绘图展示的一个包
import numpy as np

img = cv2.imread('Resources/car.png',cv2.IMREAD_GRAYSCALE) #读取一张图片 并进行了灰度变换

v1=cv2.Canny(img,80,150) #80 minVal越小 拿到的边界希望就越大，信息相对来说就更少 150 maxVal越大 拿到的边界点就越多 信息就相对更少
v2=cv2.Canny(img,50,100)

res = np.hstack((v1,v2))
cv2.imshow('res image',res)

cv2.imshow('image',img) #两个参数 第一个参数是显示图像的名字 第二个参数是读取的图像

cv2.waitKey(0) #0代表图像一直显示 1000表示是1000ms 延时1s进行自动关闭
cv2.destroyAllWindows() #触发一些按键可以执行关闭