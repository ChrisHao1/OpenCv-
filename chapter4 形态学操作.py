################ 腐蚀操作 #################
import cv2
import matplotlib.pyplot as plt #绘图展示的一个包
import numpy as np
kernel1 = np.ones((3,3),np.uint8)
kernel2 = np.ones((30,30),np.uint8)

img1 = cv2.imread('Resources/dige.png') #读取一张图片
erosion = cv2.erode(img1,kernel1,iterations = 1) #kernel卷积核 iterations = 1迭代次数

img2 = cv2.imread("Resources/pie.png")

erosion_1 = cv2.erode(img2,kernel2,iterations = 1)
erosion_2 = cv2.erode(img2,kernel2,iterations = 2)
erosion_3 = cv2.erode(img2,kernel2,iterations = 3) #进行3此腐蚀 腐蚀次数越多 图像会显得更小
res = np.hstack((erosion_1,erosion_2,erosion_3)) #横向拼接在一起

cv2.imshow("image",img1)
cv2.imshow("erosion image",erosion)
cv2.imshow("erosion_1 image",erosion_1)
cv2.imshow("erosion_2 image",erosion_2)
cv2.imshow("erosion_3 image",erosion_3)
cv2.imshow("res image",res)

cv2.waitKey(0) #0代表图像一直显示 1000表示是1000ms 延时1s进行自动关闭
cv2.destroyAllWindows() #触发一些按键可以执行关闭



################# 膨胀操作 #################
#膨胀操作与腐蚀操作互为逆运算
import cv2
import numpy as np

kernel = np.ones((5,5),np.uint8)

img = cv2.imread('Resources/dige.png')
erosion_img = cv2.erode(img,kernel,iterations = 1) #kernel卷积核 iterations = 1迭代次数
dilate_img1 = cv2.dilate(erosion_img,kernel,iterations = 1) #膨胀操作
dilate_img2 = cv2.dilate(erosion_img,kernel,iterations = 2)
dilate_img3 = cv2.dilate(erosion_img,kernel,iterations = 3)

res = np.hstack((dilate_img1,dilate_img2,dilate_img3)) #横向拼接在一起

cv2.imshow("image",img)
cv2.imshow("erosion_image",erosion_img)
cv2.imshow('dilate1 iamge',dilate_img1)
cv2.imshow('dilate2 iamge',dilate_img2)
cv2.imshow('dilate3 iamge',dilate_img3)
cv2.imshow('res image',res)

cv2.waitKey(0) #0代表图像一直显示 1000表示是1000ms 延时1s进行自动关闭
cv2.destroyAllWindows() #触发一些按键可以执行关闭



################# 开运算与闭运算 #################
#开运算 先腐蚀 后膨胀 比如说加载进来迪哥这张图片 有一些毛刺的地方 先把毛刺的地方去掉 这过程会导致原来的图像一定比例的缩小 然后再将图像进行放大 回复到原来的大小
import cv2
import numpy as np

kernel = np.ones((5,5),np.uint8)

img = cv2.imread('Resources/dige.png')
opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel) #cv2.morphologyEx形态学操作 cv2.MORPH_OPEN开运算操作

cv2.imshow('opening image',opening)

cv2.waitKey(0)
cv2.destroyAllwindowns()



#闭运算 先膨胀 再腐蚀 这样图像的一些毛刺 并不能将它去掉 反而还会放大毛刺
closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)



################# 梯度运算 #################
#梯度=膨胀-腐蚀 先把原始图像进行膨胀 再将原始图像进行腐蚀 再将膨胀的结果和腐蚀的结果进行减法操作 可以得到一个轮廓 边界信息
import cv2
import numpy as np

kernel = np.ones((5,5),np.uint8)

img = cv2.imread('Resources/pie.png')
dilate = cv2.dilate(img,kernel,iterations = 5)
erosion = cv2.erode(img,kernel,iterations = 5)
gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel) #morphologyEx形态学

res = np.hstack((dilate,erosion))

cv2.imshow('res', res)
cv2.imshow('gradient image', gradient)

cv2.waitKey(0)
cv2.destroyAllWindows()



################# 礼帽与黑帽 #################
#礼帽 = 原始输入-开运算结果 这里我们开运算的结果是我们把图像的毛刺去掉了 原始图像是带有毛刺的 所以减掉 就是得到只有刺的图像
import cv2
import numpy as np

kernel = np.ones((5,5),np.uint8)
img = cv2.imread('Resources/dige.png')
tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)

cv2.imshow('tophat image', tophat)
cv2.waitKey(0)
cv2.destroyAllWindows()



#黑帽 = 闭运算-原始输入 闭运算保留图片中的刺 原始图像中也有刺 所以相减会剩下一个带点的轮廓
import cv2
import numpy as np

kernel = np.ones((5,5),np.uint8)
img = cv2.imread('Resources/dige.png')
blackhat  = cv2.morphologyEx(img,cv2.MORPH_BLACKHAT, kernel)

cv2.imshow('blackhat image', blackhat )
cv2.waitKey(0)
cv2.destroyAllWindows()
