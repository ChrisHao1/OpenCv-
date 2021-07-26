#RGB图像的颜色通道 opencv读取图像的默认格式是BGR 黑白图没有颜色通道 灰度图有一个颜色通道
#图像在计算机中是由很多个像素点组成的

################# 读取图像 #################
import cv2
import matplotlib.pyplot as plt #绘图展示的一个包
import numpy as np

img = cv2.imread('Resources/shapes.png') #读取一张图片

imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) #进行灰度处理

cv2.imshow('image',img) #两个参数 第一个参数是显示图像的名字 第二个参数是读取的图像
cv2.imshow("Gray Image",imgGray) #将灰度图像显示出来

#'test.jpg'存储路径，一般存储在venv下  img图像变量  [int(cv2.IMWRITE_JPEG_QUALITY),70]存盘标识 图片品质为70
#cv2.CV_IMWRITE_JPEG_QUALITY  设置图片格式为.jpeg或者.jpg的图片质量，其值为0---100（数值越大质量越高），默认95
#cv2.CV_IMWRITE_WEBP_QUALITY  设置图片的格式为.webp格式的图片质量，值为0--100
#cv2.CV_IMWRITE_PNG_COMPRESSION  设置.png格式的压缩比，其值为0--9（数值越大，压缩比越大），默认为3
cv2.imwrite('test.jpg',img,[int(cv2.IMWRITE_JPEG_QUALITY),70]) #保存图片

print(img.shape) #获取图像的w h c w为宽度 h为高度 c代表有几通道 这里读出来的是3 代表彩色图
print(imgGray.shape) #灰度图只有一个颜色通道 只代表大小
print(type(img)) #这张图像的类型
print(img.size) #图像的像素点个数
print(img.dtype) #图像数据的类型
cv2.waitKey(0) #0代表图像一直显示 1000表示是1000ms 延时1s进行自动关闭
cv2.destroyAllWindows() #触发一些按键可以执行关闭



################# 读取视频 摄像头 #################
#视频由图像组成 有很多帧 每一帧都是静止的图像 我们把静止的图像连在一起 形成当前看的动的视频
import cv2
import numpy as np
#读取视频
vc = cv2.VideoCapture('Resources/test_video.mp4')
#判断是否打开视频  vc.read()一直循环读取每一帧   open,frame返回两个值  open为布尔类型true或者false frame显示当前图像
if vc.isOpened():
    open,frame = vc.read()
else:
    open = False

#遍历每一帧
while open:
    ret,frame = vc.read()
    if frame is None:
        break
    if ret == True:
        imgGray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) #进行灰度处理
        cv2.imshow("Image", frame)  #这里是对每一帧数进行处理的所以是frame
        cv2.imshow("Gray Image",imgGray)
        if cv2.waitKey(100) & 0xff == 27:  #10处理视频里的每一帧 越大就处理越慢  27按ese可以退出
            break

vc.release()
cv2.destroyAllWindows()



################# roi区域 选择我们感兴趣的区域 #################
#ROI(Region of Interest)是指图像中的一个矩形区域，可能你后续的程序需要单独处理这一个小区域 ROI一定在图像内部，而不能超出图像的范围
#如果你对图像设置了ROI，那么，Opencv的大多数函数只在该ROI区域内运算（只处理该ROI区域），如果没设ROI的话，就会出来整幅图像。
import cv2
import numpy as np
import matplotlib.pyplot as plt #绘图展示的一个包

img1 = cv2.imread('Resources/shapes.png')
#进行灰度处理
imgGray1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
imgGray2 = imgGray1[0:200,0:200] #对图像进行裁剪 选择切片
b, g, r = cv2.split(img1) #一个彩色图片有B G R三个通道组成,我们可以把三通道的数据提取出来
img2 = cv2.merge((b, g, r))  #B G R合并起来再次合并成彩色图像

cv2.imshow('image1',img1)
cv2.imshow("Gray Image1",imgGray1)
cv2.imshow("Gray Image2",imgGray2)
cv2.imshow('image2',img2)

#只保留R通道
cur_img = img1.copy()
cur_img[:, :, 0] = 0
cur_img[:, :, 1] = 0
cv2.imshow('R', cur_img)

#只保留G通道
cur_img = img1.copy()
cur_img[:, :, 0] = 0
cur_img[:, :, 2] = 0
cv2.imshow('G', cur_img)

#只保留B通道
cur_img = img1.copy()
cur_img[:, :, 1] = 0
cur_img[:, :, 2] = 0
cv2.imshow('B', cur_img)
#将BGR综合在一起时就是为我们日常所见的图像
cv2.waitKey(0)
cv2.destroyAllWindows() #触发任意键时就把窗口关闭掉



################# 边界填充 #################
import cv2
import matplotlib.pyplot as plt #绘图展示的一个包
import numpy as np

img = cv2.imread('Resources/shapes.png') #读取一张图片
top_size,bottom_size,left_size,right_size = (50,50,50,50) #上下左右进行边界填充

#BORDER_REPLICATE：复制法，也就是复制最边缘像素。
replicate = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, borderType=cv2.BORDER_REPLICATE)
#BORDER_REFLECT：反射法，对感兴趣的图像中的像素在两边进行复制例如：fedcba|abcdefgh|hgfedcb
reflect = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size,cv2.BORDER_REFLECT)
#BORDER_REFLECT_101：反射法，也就是以最边缘像素为轴，对称，gfedcb|abcdefgh|gfedcba
reflect101 = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, cv2.BORDER_REFLECT_101)
#BORDER_WRAP：外包装法cdefgh|abcdefgh|abcdefg
wrap = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size, cv2.BORDER_WRAP)
#BORDER_CONSTANT：常量法，常数值填充。
constant = cv2.copyMakeBorder(img, top_size, bottom_size, left_size, right_size,cv2.BORDER_CONSTANT, value=0)

#展示填充效果
plt.subplot(231), plt.imshow(img, 'gray'), plt.title('ORIGINAL')
plt.subplot(232), plt.imshow(replicate, 'gray'), plt.title('REPLICATE')
plt.subplot(233), plt.imshow(reflect, 'gray'), plt.title('REFLECT')
plt.subplot(234), plt.imshow(reflect101, 'gray'), plt.title('REFLECT_101')
plt.subplot(235), plt.imshow(wrap, 'gray'), plt.title('WRAP')
plt.subplot(236), plt.imshow(constant, 'gray'), plt.title('CONSTANT')
plt.show()

cv2.waitKey(0) #0代表图像一直显示 1000表示是1000ms 延时1s进行自动关闭
cv2.destroyAllWindows() #触发一些按键可以执行关闭



################# 数值计算 #################
import cv2
import matplotlib.pyplot as plt #绘图展示的一个包
import numpy as np

img1 = cv2.imread('Resources/shapes.png') #读取一张图片
img2 = img1 + 10 #在图像中 在每一个像素点上都加上10

cv2.imshow('image1', img1)
cv2.imshow('image2', img2)


cv2.waitKey(0) #0代表图像一直显示 1000表示是1000ms 延时1s进行自动关闭
cv2.destroyAllWindows() #触发一些按键可以执行关闭



################# 图像融合 #################
#把两张图像融合在一起 要保证两张图像shape相同
import cv2
import matplotlib.pyplot as plt #绘图展示的一个包
import numpy as np

img1 = cv2.imread('Resources/dog.jpg') #读取一张图片
img2 = cv2.imread('Resources/cat.jpg')
img3 = cv2.resize(img1,(500,414)) #改变目标图片的大小 这里使得狗的图片变成和猫图片一样的shape 实际写得宽与高与显示出来的是反的
img4 = cv2.addWeighted(img2, 0.4, img3, 0.6, 0) #进行图像的融合 addWeighted添加权重项 img2第一张图片输入 img3第二张图片输入 0.4权重值 0.6权重值 0偏置项提亮多少亮度

img5 = cv2.resize(img1, (0, 0), fx=3, fy=1) #fx=3横向拉长3倍 fy=1纵向不变
img6 = cv2.resize(img1, (0, 0), fx=1, fy=3)

print(img1.shape) #求得狗的shape值
print(img2.shape) #求得猫的shape值
print(img3.shape)

cv2.imshow('image1', img1)
cv2.imshow('image2', img2)
cv2.imshow('image4', img4)
cv2.imshow('image5', img5)
cv2.imshow('image6', img6)

cv2.waitKey(0) #0代表图像一直显示 1000表示是1000ms 延时1s进行自动关闭
cv2.destroyAllWindows() #触发一些按键可以执行关闭