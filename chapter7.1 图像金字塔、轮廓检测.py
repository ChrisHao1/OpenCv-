################ 高斯金字塔 #################
import cv2
import matplotlib.pyplot as plt #绘图展示的一个包
import numpy as np

img = cv2.imread('Resources/AM.png',cv2.IMREAD_GRAYSCALE) #读取一张图片 并进行了灰度变换
up1 = cv2.pyrUp(img) #将图像进行向上采样 扩大图像
up2 = cv2.pyrUp(up1) #将图像进行二次向上采样
down = cv2.pyrDown(img) #将图像进行向下采样 缩小图像

print(img.shape)
print(up1.shape)
print(up2.shape)
print(down.shape)

cv2.imshow('image',img)
cv2.imshow('up1 image',up1)
cv2.imshow('up2 image',up2)
cv2.imshow('down image',down)


cv2.waitKey(0) #0代表图像一直显示 1000表示是1000ms 延时1s进行自动关闭
cv2.destroyAllWindows() #触发一些按键可以执行关闭



#将原始图像和经过向上金字塔和向下金字塔的图像进行对比
import cv2
import matplotlib.pyplot as plt #绘图展示的一个包
import numpy as np

img = cv2.imread('Resources/AM.png',cv2.IMREAD_GRAYSCALE) #读取一张图片 并进行了灰度变换
up = cv2.pyrUp(img) #将图像进行向上采样 扩大图像
down = cv2.pyrDown(up) #将图像进行向下采样 缩小图像

print(img.shape)
print(down.shape)

res = np.hstack((img,down)) #进行横向的拼接图像

cv2.imshow('res',res)

cv2.waitKey(0) #0代表图像一直显示 1000表示是1000ms 延时1s进行自动关闭
cv2.destroyAllWindows() #触发一些按键可以执行关闭



################ 拉普拉斯金字塔 #################
import cv2
import numpy as np

img1 = cv2.imread('Resources/AM.png')
down = cv2.pyrDown(img1) #进行第一次向上采样
down_up = cv2.pyrUp(down) #在向上采样点基础上进行向下采样
img2 = img1 - down_up #进行拉普拉斯运算

res = np.hstack((img1,img2)) #进行横向的拼接图像

cv2.imshow('res',res)

cv2.waitKey(0) #0代表图像一直显示 1000表示是1000ms 延时1s进行自动关闭
cv2.destroyAllWindows() #触发一些按键可以执行关闭



################ 轮廓检测 #################
import cv2
import numpy as np

img = cv2.imread('Resources/contours.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #进行灰度处理
ret,thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY) #二值处理 亮的地方全为白 255代表白 暗的地方都为黑 0代表黑 小于127的取0 大于127的取255

binary, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
#thresh二值处理完的图像 binary第一个返回值就是二值的结果 contours保留轮廓的信息 hierarchy保留层级信息

draw_img = img.copy() #把原图像进行复制
res = cv2.drawContours(draw_img, contours, -1, (0, 0, 255), 2) #draw_img选择绘画的图像 contours轮廓信息 -1把所有轮廓画出来 (0, 0, 255)颜色模式 2线条宽度

cv2.imshow('res',res)

cv2.waitKey(0) #0代表图像一直显示 1000表示是1000ms 延时1s进行自动关闭
cv2.destroyAllWindows() #触发一些按键可以执行关闭



#轮廓特征
import cv2
import numpy as np

img = cv2.imread('Resources/contours.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #进行灰度处理
ret,thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY) #二值处理 亮的地方全为白 255代表白 暗的地方都为黑 0代表黑 小于127的取0 大于127的取255

binary, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
#thresh二值处理完的图像 binary第一个返回值就是二值的结果 contours保留轮廓的信息 hierarchy保留层级信息

draw_img = img.copy() #把原图像进行复制
res = cv2.drawContours(draw_img, contours, -1, (0, 0, 255), 2) #draw_img选择绘画的图像 contours轮廓信息 -1把所有轮廓画出来 (0, 0, 255)颜色模式 2线条宽度

cnt = contours[0] #把具体轮廓拿出来 0代表第1个轮廓
print(cv2.contourArea(cnt)) #计算轮廓面积
print(cv2.arcLength(cnt,True)) #计算轮廓周长，True表示闭合的

cv2.imshow('res',res)

cv2.waitKey(0) #0代表图像一直显示 1000表示是1000ms 延时1s进行自动关闭
cv2.destroyAllWindows() #触发一些按键可以执行关闭



################ 轮廓近似 #################
import cv2
import numpy as np

img = cv2.imread('Resources/contours2.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #进行灰度处理
ret,thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY) #二值处理 亮的地方全为白 255代表白 暗的地方都为黑 0代表黑 小于127的取0 大于127的取255

binary, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
#thresh二值处理完的图像 binary第一个返回值就是二值的结果 contours保留轮廓的信息 hierarchy保留层级信息

cnt = contours[0]

epsilon = 0.16*cv2.arcLength(cnt,True) #按照周长的百分比进行设置 0.15这个值越小越接近轮廓本身
approx = cv2.approxPolyDP(cnt,epsilon,True)

draw_img = img.copy()
res = cv2.drawContours(draw_img, [approx], -1, (0, 0, 255), 2)
cv2.imshow('res',res)



cv2.waitKey(0) #0代表图像一直显示 1000表示是1000ms 延时1s进行自动关闭
cv2.destroyAllWindows() #触发一些按键可以执行关闭



################ 画出外接矩形轮廓 #################
import cv2
import numpy as np

img = cv2.imread('Resources/contours2.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #进行灰度处理
ret,thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY) #二值处理 亮的地方全为白 255代表白 暗的地方都为黑 0代表黑 小于127的取0 大于127的取255

binary, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
#thresh二值处理完的图像 binary第一个返回值就是二值的结果 contours保留轮廓的信息 hierarchy保留层级信息

cnt = contours[0] ##把具体轮廓拿出来 0代表第1个轮廓

x,y,w,h = cv2.boundingRect(cnt) #传入轮廓信息 知道了4个坐标点
img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2) #画出矩形

cv2.imshow('img',img,)


cv2.waitKey(0) #0代表图像一直显示 1000表示是1000ms 延时1s进行自动关闭
cv2.destroyAllWindows() #触发一些按键可以执行关闭



################ 画出外接圆轮廓 #################
import cv2
import numpy as np

img = cv2.imread('Resources/contours2.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #进行灰度处理
ret,thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY) #二值处理 亮的地方全为白 255代表白 暗的地方都为黑 0代表黑 小于127的取0 大于127的取255

binary, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
#thresh二值处理完的图像 binary第一个返回值就是二值的结果 contours保留轮廓的信息 hierarchy保留层级信息

cnt = contours[0] ##把具体轮廓拿出来 0代表第1个轮廓

(x,y),radius = cv2.minEnclosingCircle(cnt)
center = (int(x),int(y))
radius = int(radius)
img = cv2.circle(img,center,radius,(0,255,0),2)

cv2.imshow('img',img,)

cv2.waitKey(0) #0代表图像一直显示 1000表示是1000ms 延时1s进行自动关闭
cv2.destroyAllWindows() #触发一些按键可以执行关闭




