# ################ 单模板匹配一个目标 #################
# import cv2
# import numpy as np
# from matplotlib import pyplot as plt
#
# img = cv2.imread('Resources/lena.jpg')
# template = cv2.imread('Resources/lena.png')
#
# theight, twidth = template.shape[:2] #获得模板图片的高宽尺寸
#
#
# methods = ['cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED',
#             'cv2.TM_CCORR','cv2.TM_CCORR_NORMED',
#             'cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED']
#
# for meth in methods:
#     method = eval(meth)
#
#     res = cv2.matchTemplate(img,template,method) #进行模板匹配
#     min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res) #找到最大值和最小值 返回最小值 最大值 最小值坐标位置 最大值坐标位置
#     print('{}:cv2.minMaxLoc(res):\t{} '.format(meth,cv2.minMaxLoc(res)))
#
#     if method in [cv2.TM_SQDIFF,cv2.TM_SQDIFF_NORMED]: #进行判断 如果是平方差匹配cv2.TM_SQDIFF或归一化匹配cv2.TM_SQDIFF_NORMED，取最小值
#         top_left = min_loc
#     else:
#         top_left = max_loc
#
#     bottom_right = (top_left[0] + twidth, top_left[1] + theight)
#     cv2.rectangle(img,top_left, bottom_right, (0,255,0), 2)
#
#     plt.subplot(121),plt.imshow(res,cmap = 'gray')
#     plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
#     plt.subplot(122),plt.imshow(img,cmap = 'gray')
#     plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
#     plt.suptitle(meth)
#     plt.show()



################ 单模板匹配多个目标 #################
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('Resources/lena.jpg')
imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) #进行灰度处理
template = cv2.imread('Resources/lena.png')

theight, twidth = template.shape[:2] #获得模板图片的高宽尺寸

res = cv2.matchTemplate(imgGray,template,cv2.TM_CCOEFF_NORMED) #进行模板匹配

threshoid = 0.8
loc = np.where(res >= threshoid)
for pt in zip(*loc[::-1]):
    bottom_right = (pt[0] + twidth, pt[1] + theight)
    cv2.rectangle(img, pt, (pt[0] + twidth, pt[1] + theight), (0,0,255), 2)

cv2.imshow('image',img)
cv2.waitKey(0)


