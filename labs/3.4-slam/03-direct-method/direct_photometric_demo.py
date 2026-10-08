# DSO 直接法
# 两个简单图片
#       ↓
# 选择一个像素
#       ↓
# 假设深度
#       ↓
# 假设相机移动
#       ↓
# 3D 点
#       ↓
# 投影到下一帧
#       ↓
# 读取两个位置的亮度
#       ↓
# 计算光度误差

from pathlib import Path

import cv2
import numpy as np

#当前脚本所在目录
base_dir=Path(__file__).parent

#两张图片 01下面
image1_path=(
    base_dir.parent
    /"01-feature-matching"
    /"images"
    /"0001.JPG"
)

image2_path=(
    base_dir.parent
    /"01-feature-matching"
    /"images"
    /"0002.JPG"
)








# 第一部分：读取两张图片→ 转成灰度图→ 选择第一张图中的一个像素→ 打印这个像素在两帧中的亮度值
#读取图片
def read_image_unicode(path):
    """
    使用 NumPy 读取文件字节，再用 OpenCV 解码。
    这样可以避免 Windows 中文路径下 imread 失败。
    """

    file_bytes=np.fromfile(str(path),dtype=np.uint8)
    image=cv2.imdecode(file_bytes,cv2.IMREAD_COLOR)

    if image is None:
        raise FileNotFoundError(f"图片读取失败:{path}")

    return image

#缩放图片
def resize_keep_ratio(image,max_side=1280):
    """
    按比例缩小图片，使最长边不超过 max_side。
    """
    height,width=image.shape[:2]
    scale=min(1.0,max_side/max(height,width))

    if scale<1.0:
        new_width=int(width*scale)
        new_height=int(height*scale)

        return cv2.resize(
            image,
            (new_width,new_height),
            interpolation=cv2.INTER_AREA
        )

    return image


#读取并缩放两张图片
img1=resize_keep_ratio(read_image_unicode(image1_path))
img2=resize_keep_ratio(read_image_unicode(image2_path))

#直接法通常使用灰度亮度，所以先转成灰度图
gray1=cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
gray2=cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)

#获取灰度图尺寸
height,width=gray1.shape

#在第一张图中选择一个像素
#这里先选图片中心附近的像素
#u：图像的列坐标，对应相机坐标里的 x 方向
#v：图像的行坐标，对应相机坐标里的 y 方向
#u 是横向   v 是纵向
# u是横向,对应x，所以x_c用u，
# v是纵向,对应y，所以y_c用v
# 然后u是横向，v是纵向，像素是行，列，所以对应是v，u
u=width//2
v=height//2

#读取同一个位置在两帧中的亮度
# 图像数组的顺序是：gray[高, 宽]
# 也就是：第一个索引：v，行,第二个索引：u，列
brightness1=float(gray1[v,u])
brightness2=float(gray2[v,u])

print("gray1.shape =",gray1.shape)
print("gray2.shape =",gray2.shape)

print("选中像素位置(u,v) =",(u,v))
print("第一帧亮度 =",brightness1)
print("第二帧同一位置亮度 =",brightness2)









#  第二部分  假设像素深度，并把它反投影成相机坐标中的三维点
# 近似相机内参
# 真实相机应该通过标定得到
focal=1.0*max(width,height)

K=np.array(
    [
        [focal,0.0,width/2.0],
        [0.0,focal,height/2.0],
        [0.0,0.0,1.0]
    ],
    dtype=float
)

#从K中取出参数
fx=K[0,0]
fy=K[1,1]
cx=K[0,2]
cy=K[1,2]

print("K =")
print(K)

#假设当前像素的深度
#这个值不是从图像中得到的，而是人为假设的
depth=10.0

#像素->相机坐标射线
x_c=(u-cx)/fx
y_c=(v-cy)/fy

ray_c=np.array( 
    [
        x_c,
        y_c,
        1.0
    ],
    dtype=float
)

#射线+深度 ->相机坐标三维点
point_c=ray_c*depth

print("ray_c =")
print(ray_c)

print("point_c =")
print(point_c)

print("point_c.shape =",point_c.shape) #3,











# 第 3 部分：假设相机移动 R、t，把三维点投影到第二帧，得到预测像素位置
# 先假设一个简单运动：
# R = 单位矩阵，表示两帧之间暂时没有旋转
# t = [-0.2, 0, 0]，表示相机沿自己的 x 方向移动一点
# 假设第一帧到第二帧之间存在相对运动
# 这里先人为设置，后面真实 VO 会通过匹配点估计 R、t
R=np.eye(3)

t=np.array(
    [
        -0.2,
        0.0,
        0.0
    ],
    dtype=float
)

#把第一帧相机坐标中的三维点
#转换到第二帧相机坐标
point_c2= R @ point_c + t

print("point_c2 =")
print(point_c2)


#把第二帧相机坐标点投影成像素
projected_h2=K @ point_c2

print("projected_h2 =")
print(projected_h2)

#透视除法：除以第三个分量
u2=projected_h2[0]/projected_h2[2]
v2=projected_h2[1]/projected_h2[2]

predicted_pixel=np.array(
    [
        u2,
        v2
    ],
    dtype=float
)

print("第一帧像素位置(u,v) =",(u,v))
print("第二帧预测像素位置(u2,v2) =",predicted_pixel)










# 第 4 部分：读取第二帧中预测像素位置的亮度,计算光度误差
# 预测位置是小数，先取整成整数像素索引
u2_int = int(round(u2))
v2_int = int(round(v2))
print("取整后的预测像素 =",(u2_int,v2_int))

#检查预测像素是否还在图片范围内
if 0<= u2_int <width and 0<= v2_int <height:
    #图像数组索引顺序是[行，列]=[v,u]
    brightness2_predicted=float(gray2[v2_int,u2_int])
else:
    raise ValueError(
        f"预测像素超出图像范围:{(u2_int,v2_int)}"
    )

#计算第一帧和预测位置之间的广度误差
photometric_error=abs(
    brightness1-brightness2_predicted
)

print("第一帧像素亮度 =",brightness1)
print("第二帧同一位置亮度 =",brightness2)
print("第二帧预测位置 =",(u2_int,v2_int))
print("第二帧预测位置亮度 =",brightness2_predicted)
print("光度误差 =",photometric_error)

