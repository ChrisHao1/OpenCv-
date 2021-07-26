################# 图像阈值处理 #################
#ret,dst = cv2.threshold(src, thresh, maxval, type)
# src： 输入图，只能输入单通道图像，通常来说为灰度图
# dst： 输出图
# thresh： 阈值 常常为127为界进行判断 图像像素点范围为0~255
# maxval： 通常为255 当像素值超过了阈值（或者小于阈值，根据type来决定），所赋予的值
# type：二值化操作的类型，选择的方法，功能 包含以下5种类型： cv2.THRESH_BINARY； cv2.THRESH_BINARY_INV； cv2.THRESH_TRUNC； cv2.THRESH_TOZERO；cv2.THRESH_TOZERO_INV
# cv2.THRESH_BINARY 超过阈值部分取maxval（最大值），否则取0 大于阈值取一个值 小于阈值取一个值 假设thresh为127 maxval为255 对每一个像素点进行判断大于127取255 小于127取0 一般图像中越亮的地方值越大
# cv2.THRESH_BINARY_INV THRESH_BINARY的反转
# cv2.THRESH_TRUNC 大于阈值部分设为阈值，否则不变
# cv2.THRESH_TOZERO 大于阈值部分不改变，否则设为0
# cv2.THRESH_TOZERO_INV THRESH_TOZERO的反转

import cv2
import matplotlib.pyplot as plt #绘图展示的一个包
import numpy as np

img = cv2.imread('Resources/cat.jpg') #读取一张图片
imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) #进行灰度处理

ret,thresh1 = cv2.threshold(imgGray, 127, 255, cv2.THRESH_BINARY) #亮的地方全为白 暗的地方都为黑 小于127的取0 大于127的取255
ret,thresh2 = cv2.threshold(imgGray, 127, 255, cv2.THRESH_BINARY_INV) #亮的地方全为黑 暗的地方都为白 _INV一种反转的表示方法 小于127的取255 大于127的取0
ret,thresh3 = cv2.threshold(imgGray, 127, 255, cv2.THRESH_TRUNC) #截断 127为一个临界点 大于127的等于127 小于127不进行改变
ret,thresh4 = cv2.threshold(imgGray, 127, 255, cv2.THRESH_TOZERO) #大于127的保持不变 小于等于127为0 相当于图像中比较亮的地方不进行变化 暗的地方全为黑点
ret,thresh5 = cv2.threshold(imgGray, 127, 255, cv2.THRESH_TOZERO_INV) #大于127全为0 小于等于127为保持不变 相当于图像中比较亮的敌方变为黑点 暗的地方保持不变

cv2.imshow('THRESH_BINARY',thresh1)
cv2.imshow('THRESH_BINARY_INV',thresh2)
cv2.imshow('THRESH_TRUNC',thresh3)
cv2.imshow('THRESH_TOZERO',thresh4)
cv2.imshow('THRESH_TOZERO_INV',thresh5)

cv2.waitKey(0) #0代表图像一直显示 1000表示是1000ms 延时1s进行自动关闭
cv2.destroyAllWindows() #触发一些按键可以执行关闭



################# 图像平滑处理 #################
#对图像进行滤波操作
import cv2
import matplotlib.pyplot as plt #绘图展示的一个包
import numpy as np

img = cv2.imread('Resources/lenaNoise.png') #输入的图像上面有一些噪音点

blur = cv2.blur(img, (5, 5)) #均值滤波 简单的平均卷积操作 (5, 5)内核大小 一般为奇数 在图像中构造一个卷积矩阵 5×5的各行各列都为1的矩阵 选取25个数 进行平均处理（除以25）
box1 = cv2.boxFilter(img,-1,(3,3), normalize=True) #方框滤波 基本和均值一样，可以选择归一化 -1这个数据一般不需要改 颜色通道上一致 (3,3)卷积核 normalize=True这时候等于均值滤波
box2 = cv2.boxFilter(img,-1,(3,3), normalize=False) #方框滤波 normalize=False不做平均处理 超过255的地方直接等于255 显示为白色
aussian = cv2.GaussianBlur(img, (5, 5), 1) # 高斯滤波 高斯模糊的卷积核里的数值是满足高斯分布，相当于更重视中间的 离的越近的发挥效果应当越大 离的越远的发挥效果比较小 自己构造一个权重矩阵
median = cv2.medianBlur(img, 5) #中值滤波 相当于用中值代替 假设矩阵为5×5 在这9个数中 选取中间的数 作为处理结果 一般可以采取这种方法进行去除噪音点

res1 = np.hstack((blur,box1,box2,aussian,median)) #图像横向拼接在一起
res2 = np.vstack((blur,box1,box2,aussian,median)) #图像纵向拼接在一起

cv2.imshow('image',img)
cv2.imshow('blur image',blur)
cv2.imshow('box1 image',box1)
cv2.imshow('box2 image',box2)
cv2.imshow('aussian image',aussian)
cv2.imshow('median image',median)
cv2.imshow('res1 image',res1)
cv2.imshow('res2 image',res2)

cv2.waitKey(0) #0代表图像一直显示 1000表示是1000ms 延时1s进行自动关闭
cv2.destroyAllWindows() #触发一些按键可以执行关闭


