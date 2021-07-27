################ Sobel算子 #################
#进行水平方向的计算 从右往左
import cv2
import numpy as np

img = cv2.imread('Resources/pie.png')
sobelx1 = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=3) #cv2.CV_64F 白到黑是正数，黑到白是负数，所有的负数都会被截断成0  1,0 1代表算水平进行运算 0代表竖直不参与运算
sobelx2 = cv2.convertScaleAbs(sobelx1) #进行让负数取绝对值的操作 都是变为正数

sobely1 = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=3) #cv2.CV_64F 白到黑是正数，黑到白是负数，所有的负数都会被截断成0  0,1 0代表算水平不参与运算 1代表竖直参与运算
sobely2 = cv2.convertScaleAbs(sobelx1) #进行让负数取绝对值的操作 都是变为正数

cv2.imshow('image',img)
cv2.imshow('sobelx1 image',sobelx1)
cv2.imshow('sobelx2 image',sobelx2)

cv2.waitKey(0)
cv2.destroyAllwindowns()



#进行竖直方向的计算 从下往上
import cv2
import numpy as np

img = cv2.imread('Resources/pie.png')

sobely1 = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=3) #cv2.CV_64F 白到黑是正数，黑到白是负数，所有的负数都会被截断成0  0,1 0代表算水平不参与运算 1代表竖直参与运算
sobely2 = cv2.convertScaleAbs(sobely1) #进行让负数取绝对值的操作 都是变为正数

cv2.imshow('image',img)
cv2.imshow('sobely1 image',sobely1)
cv2.imshow('sobely2 image',sobely2)

cv2.waitKey(0)
cv2.destroyAllwindowns()



#进行水平和竖直方向求和的计算
import cv2
import numpy as np

img = cv2.imread('Resources/pie.png')

sobelx1 = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=3) #cv2.CV_64F 白到黑是正数，黑到白是负数，所有的负数都会被截断成0  1,0 1代表算水平进行运算 0代表竖直不参与运算
sobelx2 = cv2.convertScaleAbs(sobelx1) #进行让负数取绝对值的操作 都是变为正数

sobely1 = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=3) #cv2.CV_64F 白到黑是正数，黑到白是负数，所有的负数都会被截断成0  0,1 0代表算水平不参与运算 1代表竖直参与运算
sobely2 = cv2.convertScaleAbs(sobely1) #进行让负数取绝对值的操作 都是变为正数

sobelxy = cv2.addWeighted(sobelx2,0.5,sobely2,0.5,0) #0.5为权重 0偏置项

res = np.hstack((sobelx2,sobely2,sobelxy)) #横向拼接在一起

cv2.imshow('res image',res)

cv2.waitKey(0)
cv2.destroyAllwindowns()



#同时及逆行Gx和Gy的运算
import cv2
import numpy as np

img = cv2.imread('Resources/pie.png')

sobelx1 = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=3) #cv2.CV_64F 白到黑是正数，黑到白是负数，所有的负数都会被截断成0  1,0 1代表算水平进行运算 0代表竖直不参与运算
sobelx2 = cv2.convertScaleAbs(sobelx1) #进行让负数取绝对值的操作 都是变为正数

sobely1 = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=3) #cv2.CV_64F 白到黑是正数，黑到白是负数，所有的负数都会被截断成0  0,1 0代表算水平不参与运算 1代表竖直参与运算
sobely2 = cv2.convertScaleAbs(sobely1) #进行让负数取绝对值的操作 都是变为正数

sobelxy=cv2.Sobel(img,cv2.CV_64F,1,1,ksize=3)
sobelxy1 = cv2.convertScaleAbs(sobelxy)

sobelxy2 = cv2.addWeighted(sobelx2,0.5,sobely2,0.5,0) #0.5为权重 0偏置项

res = np.hstack((sobelxy1,sobelxy2)) #横向拼接在一起

cv2.imshow('res image',res)

cv2.waitKey(0)
cv2.destroyAllwindowns()



################ Scharr和laplacian算子 #################
import cv2
import numpy as np

img = cv2.imread('Resources/lena.jpg')

#进行Sobel算子
sobelx1 = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=3)
sobely1 = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=3)
sobelx2 = cv2.convertScaleAbs(sobelx1)
sobely2 = cv2.convertScaleAbs(sobely1)
sobelxy =  cv2.addWeighted(sobelx2,0.5,sobely2,0.5,0)

#进行Scharr算子
scharrx = cv2.Scharr(img,cv2.CV_64F,1,0)
scharry = cv2.Scharr(img,cv2.CV_64F,0,1)
scharrx = cv2.convertScaleAbs(scharrx)
scharry = cv2.convertScaleAbs(scharry)
scharrxy =  cv2.addWeighted(scharrx,0.5,scharry,0.5,0)

#进行laplacian算子
laplacian1 = cv2.Laplacian(img,cv2.CV_64F) #这里没有x y 中间点与其他点进行比较
laplacian2 = cv2.convertScaleAbs(laplacian1) #把负值转换成正正值

res = np.hstack((img,sobelxy,scharrxy,laplacian2))
cv2.imshow('res',res)

cv2.waitKey(0)
cv2.destroyAllwindowns()






