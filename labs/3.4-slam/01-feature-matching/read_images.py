from pathlib import Path

import cv2
import numpy as np




# 第 1 步：准备两张图片，并让 OpenCV 正确读进来

#当前脚本所在目录
# Path(__file__)：当前脚本的路径。
# .parent：它所在的文件夹
base_dir=Path(__file__).parent


#两张图片的路径  /：Path 对象的路径拼接
img1_path=base_dir/"images"/"0001.JPG"
img2_path=base_dir/"images"/"0002.JPG"


# 定义一个函数：读取图片
def read_image_unicode(path):
    """
    使用 NumPy 先读取文件字节，再用 OpenCV 解码。

    目的：
    - 避免 OpenCV 在 Windows 中文路径下直接 imread 失败；
    - 保留图片的原始像素内容。
    """
    # 第一步：用 NumPy 读取文件字节
    # np.fromfile：把文件读成一串字节。
    # dtype=np.uint8：每个字节是 0–255 的整数。
    # str(path)：把 Path 对象转成字符串，因为 np.fromfile 需要字符串。
    # 效果：把图片文件原封不动地读进内存，变成一串数字
    file_bytes=np.fromfile(str(path),dtype=np.uint8)

    # 第二步：用 OpenCV 解码
    # cv2.imdecode：把字节解码成图像。
    # cv2.IMREAD_COLOR：读成彩色图。
    # 效果：把字节变成真正的图像数组
    image=cv2.imdecode(file_bytes,cv2.IMREAD_COLOR)

    # 第三步：检查是否成功
    if image is None:
        raise FileNotFoundError(f"图片读取失败:{path}")

    # 为什么不用 cv2.imread？
    # 因为：OpenCV 的 cv2.imread 在 Windows 中文路径下经常失败。
    # 用 np.fromfile + cv2.imdecode 可以绕过这个问题
    return image


#读取两张图片
img1=read_image_unicode(img1_path)
img2=read_image_unicode(img2_path)

#打印图片信息
#OpenCV 默认读取的是 BGR，不是 RGB  H,W,3
print("image1 path:", img1_path)
print("image1 shape:", img1.shape) #(5712, 4284, 3)
print("image1 dtype:", img1.dtype)
print("image1 min/max:", img1.min(), img1.max())
print()
print("image2 path:", img2_path)
print("image2 shape:", img2.shape) #(5712, 4284, 3)
print("image2 dtype:", img2.dtype)
print("image2 min/max:", img2.min(), img2.max())







# 第 2 步：缩小图片并转换成灰度图
# 目标
# 原始大图
#   → 按比例缩小
#   → BGR 彩色图
#   → 灰度图
# 为什么要这样做：
# - 特征检测通常只需要灰度纹理；
# - 缩小图片可以减少计算量；
# - 保持宽高比，避免图片被拉伸变形。
# 定义函数：按比例缩小图片
def resize_keep_ratio(image,max_side=1280):
    """
    按比例缩小图片，使最长边不超过 max_side。

    如果原图本来就不大，则不放大。
    """
    height,width=image.shape[:2]  #取图片的前两个维度，就是高度和宽度

    #计算缩放比例
    # max(height, width)：图片最长的那条边。
    # max_side / max(height, width)：最长边缩到 max_side 需要的比例。
    # min(1.0, ...)：如果图片本来就不大，比例不超过 1.0，也就是不放大。
    # 效果：如果图片是 4000×3000，最长边 4000，比例是 1280/4000 = 0.32。
    # 如果图片是 800×600，最长边 800，比例是 1280/800 = 1.6，但取 min(1.0, 1.6) = 1.0，不放大。
    # 生活比喻：大图 → 缩小到最长边 1280。小图 → 保持原样。
    scale=min(1.0,max_side/max(height,width))

    #如果比例小于 1.0，就缩小
    if scale<1.0:
        new_width=int(width*scale)
        new_height=int(height*scale)

        #INTER_AREA 更适合缩小图片
        resized=cv2.resize(
            image,
            (new_width,new_height),
            interpolation=cv2.INTER_AREA  #缩小图片时推荐的插值方式
        )
        return resized
    #如果图片本来就不大，直接返回原图，不做缩放
    return image


#缩小两张图片
img1_small=resize_keep_ratio(img1)
img2_small=resize_keep_ratio(img2)

#转成灰度图
#opencv 默认读入的是BGR，这里转成灰度
gray1=cv2.cvtColor(img1_small,cv2.COLOR_BGR2GRAY)
gray2=cv2.cvtColor(img2_small,cv2.COLOR_BGR2GRAY)

#查看缩小后的彩色图形状
print("img1_small shape:",img1_small.shape) #(1280, 960, 3)
print("img2_small shape:",img2_small.shape) #(1280, 960, 3)

#查看灰度图形状
print("gray1 shape:",gray1.shape) #(1280, 960)
print("gray2 shape:",gray2.shape) #(1280, 960)







# 第 3 步：检测两张图的 ORB 特征点
# 在两张灰度图里，检测 ORB 特征点，并计算它们的描述子。
# 也就是说：找出图片里“好认的点”。
# 给每个点算一个“指纹”，用来判断两张图里哪些点是同一个。

# keypoint：特征点在图片中的位置
# 1. 特征点（keypoint）
# 特征点就是：图片里好认的点。
# 比如：拐角,边缘交点,纹理明显的地方
# 每个特征点有一个位置：(x, y)。

# descriptor：描述这个特征点周围长什么样的向量
# 2. 描述子（descriptor）
# 描述子就是：描述这个特征点周围长什么样的向量。
# 它就像特征点的“指纹”。
# 同一个真实点，在不同图片里，描述子应该很相似。
# 不同的点，描述子应该差很多。

# match：判断第一张图的特征点和第二张图的哪个特征点是同一个
# 3. 匹配（match）
# 匹配就是：判断第一张图的特征点和第二张图的哪个特征点是同一个。
# 匹配靠的就是描述子。

#创建orb特征检测器
#nfeatures 表示最多保留多少个特征点
orb=cv2.ORB_create(nfeatures=1000)

#对第一张灰度图检测特征点和描述子
# orb.detectAndCompute：检测特征点并计算描述子。
# gray1：输入的灰度图。
# None：没有掩码，表示整张图都检测
keypoints1,descriptors1=orb.detectAndCompute(gray1,None)

#对第二张灰度图检测特征点和描述子
keypoints2,descriptors2=orb.detectAndCompute(gray2,None)

print("keypoints1 数量=",len(keypoints1)) #1000
print("keypoints2 数量=",len(keypoints2)) #1000

print(
    "descriptors1 shape =",
    None if descriptors1 is None else descriptors1.shape  #1000,32
)

print(
    "descriptors2 shape =",
    None if descriptors2 is None else descriptors2.shape  #1000,32
)





# 第 4 步：把两张图的特征点匹配起来
# 先理解流程：
# 第一张图的每个描述子
#   → 在第二张图里找最相似的描述子
#   → 得到匹配关系
# "相似"怎么判断？比较两个描述子的距离。
# 距离越小，越相似。
# 距离最小的那个，就是最佳匹配。

# ORB 的描述子是二进制向量，所以使用：NORM_HAMMING：计算二进制描述子的汉明距离

#创建暴力匹配器 暴力匹配器，逐个比较。
#ORB是二进制描述子，所以距离使用汉明距离
bf=cv2.BFMatcher(cv2.NORM_HAMMING)

#每个第一张图的特征点，找第二张图中最近的2个候选匹配
# 对每个特征点找最近的 2 个候选
# raw_matches = bf.knnMatch(descriptors1, descriptors2, k=2)
# knnMatch：K 近邻匹配。
# k=2：每个描述子找最近的 2 个候选。
# 为什么要找 2 个？
# 因为后面要用 Lowe ratio test 筛选。
# 返回结果：
# raw_matches：一个列表。
# 每个元素是一个 pair，包含 2 个匹配。
raw_matches=bf.knnMatch(descriptors1,descriptors2,k=2)

#使用lowe ratio test 筛选更可靠的匹配
good_matches=[]

for pair in raw_matches:
    #正常情况下pair应该有两个元素:
    #第一步：检查 pair 是否有 2 个元素
    if len(pair)<2:
        continue

    #第二步：取出最佳和次佳匹配
    best_match,second_match=pair

    #最佳匹配明显优于第二匹配时，才认为它可靠
    # 最佳匹配的距离，必须明显小于次佳匹配的距离。
    # 为什么？如果最佳和次佳差不多，说明这个匹配不够独特。
    # 如果最佳明显优于次佳，说明这个匹配很可靠。
    # 0.75 是什么？这是一个经验值。
    # 越小越严格，匹配越少但越可靠。
    # 越大越宽松，匹配越多但可能有错误。
    if best_match.distance<0.75*second_match.distance:
        good_matches.append(best_match)


print("raw matches 数量=",len(raw_matches))
print("good matches 数量=",len(good_matches))

#再画匹配结果并保存
#创建结果目录
result_dir=base_dir/"results"
result_dir.mkdir(exist_ok=True)

#画出匹配关系
matched_image=cv2.drawMatches(
    img1_small,
    keypoints1,
    img2_small,
    keypoints2,
    good_matches,
    None,  #不提供输出图像，自动创建
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS  #不画没有匹配的点
)

#使用编码方式保存，避免中文路径可能造成的问题
match_path=result_dir/"feature_matches.jpg"
success,encoded=cv2.imencode(".jpg",matched_image)  #把图像编码成字节

if not success:
    raise RuntimeError("匹配结果编码失败")

encoded.tofile(str(match_path))  #把字节写入文件。
print("匹配结果已保存:",match_path)







# good matches 里仍然有错误匹配
#   → 不能直接拿来算相机运动
#   → 需要先用几何约束筛选
#   → RANSAC
# 那些水平、方向一致、连接同一物体同一位置的线，是可靠匹配；
# 斜着穿过整张图、明显跨越不同物体的线，通常是错误匹配。

# 第 7 步：用 RANSAC 筛掉几何上不一致的匹配。
# RANSAC 会尝试找一个满足大部分匹配点的几何关系：
# 大部分正确匹配
#   → 符合同一个几何关系
#   → 保留下来

# 少数错误匹配
#   → 不符合这个关系
#   → 当作 outlier 删除

# RANSAC 的思路是：尝试找一个满足大部分匹配点的几何关系。
# 具体说：大部分正确匹配 → 符合同一个几何关系 → 保留。
# 少数错误匹配 → 不符合这个关系 → 删除。
# 这里的"几何关系"就是基础矩阵 F。

# 理解基础矩阵 F
# F 描述的是：两张图之间的对极几何关系。
# 简单说：给你第一张图上的一个点。
# 用 F 一算，就能得到第二张图上的对极线。
# 正确的匹配点，一定落在这条对极线上。
# 所以：如果一对匹配点符合 F，它可能是正确的。
# 如果不符合，它就是错误匹配。


#把good_matches 转换成两张图片中的二维点坐标
#queryIdx：这个匹配点在第一张图中的编号
#trainIdx：这个匹配点在第二张图中的编号
# m.queryIdx：这个匹配点在第一张图中的编号。
# m.trainIdx：这个匹配点在第二张图中的编号。
# keypoints1[m.queryIdx].pt：取出第一张图中这个特征点的坐标 (x, y)。
# 同理第二张图。
# 结果：
# points1：形状 (N, 2)，第一张图中的 N 个点。
# points2：形状 (N, 2)，第二张图中的 N 个点。
points1=np.float32(
    [keypoints1[m.queryIdx].pt for m in good_matches]
)
points2=np.float32(
    [keypoints2[m.trainIdx].pt for m in good_matches]
)


#用ransac 计算基础矩阵F
#1.0：像素误差阈值
#0.999：置信度
#2000：最多迭代次数

# points1, points2：两张图中的匹配点。
# cv2.FM_RANSAC：用 RANSAC 方法估计 F。
# 1.0：像素误差阈值。误差小于 1 像素的点算内点。
# 0.999：置信度。99.9% 的概率找到正确解。
# 2000：最多迭代 2000 次。
# 返回两个东西：F：基础矩阵，形状 (3, 3)。
# mask：形状 (N, 1)，标记哪些点是内点。
# mask 的含义：
# mask[i] = 1：第 i 个匹配是内点（符合 F）。
# mask[i] = 0：第 i 个匹配是外点（不符合 F）
F,mask=cv2.findFundamentalMat(
    points1,
    points2,
    cv2.FM_RANSAC,
    1.0,
    0.999,
    2000
)

if F is None:
    raise RuntimeError("基础矩阵估计失败")

#mask形状是（N，1），把它拉平成（N，）
mask=mask.ravel().astype(bool)  #转成布尔值

#只保留被ransac认为是内点的匹配
# zip(good_matches, mask)：把匹配和 mask 配对。
# if keep：只保留 mask 为 True 的匹配
inlier_matches=[
    match
    for match,keep in zip(good_matches,mask)
    if keep
]

print("RANSAC 前匹配数 =", len(good_matches))
print("RANSAC 内点数 =", len(inlier_matches))
print("RANSAC 外点数 =", len(good_matches) - len(inlier_matches))
print("基础矩阵 F shape =", F.shape)

# 画出经过 RANSAC 筛选后的匹配
filtered_image=cv2.drawMatches(
    img1_small,
    keypoints1,
    img2_small,
    keypoints2,
    inlier_matches,
    None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
)

filtered_path = result_dir / "feature_matches_ransac.jpg"
success, encoded = cv2.imencode(".jpg", filtered_image)

if not success:
    raise RuntimeError("RANSAC 匹配结果编码失败")

encoded.tofile(str(filtered_path))

print("RANSAC 过滤结果已保存:", filtered_path)







# 第 8 步：从两张图片的对应点估计相机相对运动 R、t
# 先分清两个矩阵：
# F：基础矩阵，作用在像素坐标上
# E：本质矩阵，作用在归一化相机坐标上

# 矩阵	         全称	         作用在什么坐标	         描述什么
# F	           基础矩阵	            像素坐标	      两张图的几何关系
# E	           本质矩阵	          归一化相机坐标	   两张图的几何关系
# 区别在于：
# F 直接作用在像素坐标上，不需要内参。
# E 作用在归一化相机坐标上，需要内参。
# 所以要从 F 得到相机的 R 和 t，通常要经过 E。
# 而算 E 需要相机内参 K。


# 1）先构造近似相机内参
#取出图片的高、宽
height,width=gray1.shape

#近似焦距
#真实相机应该通过标定得到焦距，这里只是为了先跑通流程
focal=1.0*max(width,height) #取最长边作为焦距

#近似内参矩阵
K_approx=np.array(
    [
        [focal,0.0,width/2.0],
        [0.0,focal,height/2.0],
        [0.0,0.0,1.0]
    ],
    dtype=float
)

print("K_approx =")
print(K_approx)
# 这里：
# fx = fy = focal  水平、垂直焦距
# cx = width / 2    主点在图片中心
# cy = height / 2   主点在图片中心
# 真实项目中，相机内参应该来自标定，而不是随意设置


# 2)只使用 RANSAC 内点估计 E
# 只用 RANSAC 认为是内点的匹配来估计 E。
# 为什么？因为 E 对错误匹配很敏感。用内点估计更准
# 取出ransac筛选后的二维对应点
points1_inliers=points1[mask]
points2_inliers=points2[mask]

#使用内点估计本质矩阵E
#ransac：继续处理少量残余错误
#1.0:像素误差阈值
#0.999：置信度
E,essential_mask=cv2.findEssentialMat(
    points1_inliers,
    points2_inliers,
    K_approx,#近似内参
    method=cv2.RANSAC, #用 RANSAC 方法估计 E
    prob=0.999,  #置信度
    threshold=1.0  #像素误差阈值
)

#如果匹配点太少或质量太差，E 可能是 None。
if E is None:
    raise RuntimeError("本质矩阵估计失败")

print("E shape =",E.shape)
print("E =")
print(E)

# 返回两个东西：E：本质矩阵，形状 (3, 3)。
# essential_mask：标记哪些点是内点。
# 为什么这里还要再用一次 RANSAC？
# 因为：即使上一步筛过一遍，仍然可能有少量残余错误匹配。
# 再用一次 RANSAC，进一步清洗。


# 3)恢复 R 和 t
#recoverPose 会从E中恢复相对旋转和平移
#返回的t只提供方向，不能提供真实物理尺度
num_pose_inliers,R,t,pose_mask=cv2.recoverPose(
    E,  #本质矩阵
    points1_inliers,  #匹配点
    points2_inliers,
    K_approx  #内参
)

# 返回四个东西：
# num_pose_inliers：内点数量。
# R：旋转矩阵，形状 (3, 3)。
# t：平移向量，形状 (3, 1)。
# pose_mask：标记哪些点支持这个解。
# 注意：t 只提供方向，不能提供真实物理尺度。
# 也就是说：你能知道相机朝哪个方向移动。但不知道移动了多少厘米/米。
# 为什么？因为从两张图片估计运动，存在尺度不确定性：
# 相机移动 1 米拍的照片。
# 相机移动 10 米拍的照片。
# 在图片上看可能是一样的。
# 所以 t 被归一化了，只有方向。

print("recoverPose 内点数 =",num_pose_inliers)
print("R shape =",R.shape)
print("t shape =",t.shape)

print("R =")
print(R)

print("t =")
print(t)

# R：第一台相机到第二台相机的旋转
# t：第一台相机到第二台相机的平移方向
# t 的长度被归一化，所以只有方向，没有真实米数
# 也就是说，你现在能得到：相机转了多大,相机朝哪个方向移动
# 但还不能直接知道：相机移动了多少厘米/米
# 因为这需要尺度信息。



# 4) 组合成 4×4 相对位姿
T_relative=np.eye(4)  #创建一个 4×4 单位矩阵
T_relative[:3,:3]=R  #前 3×3 放 R
T_relative[:3,3]=t.ravel()  #前 3 行第 4 列放 t

print("T_relative =")
print(T_relative)
#T_relative 表示：第一帧相机坐标→ 第二帧相机坐标的刚体变换。







# 第 9 步：从相对位姿中提取第二台相机的位置
# recoverPose 得到的相对位姿可以写成：P2 = R @ P1 + t
# 其中：P1：同一个点在第一帧相机坐标中的位置
# P2：同一个点在第二帧相机坐标中的位置
# 第二台相机在第一台相机坐标中的位置是：C2_in_1 = -R.T @ t
# 你可以把它理解成：
# 相机 1 在世界原点
#   → 经过 R、t
#   → 求相机 2 相对于相机 1 的位置
# why? 公式 C2_in_1 = -R.T @ t 是怎么来的？
# 我们从公式出发：P2 = R @ P1 + t
# 第一步：相机 2 的原点在相机 1 中在哪？
# 相机 2 的原点，在相机 2 自己的坐标系中是：P2 = (0, 0, 0)
# 第二步：代入公式
# 0 = R @ P1 + t
# 第三步：解出 P1
# R @ P1 = -t
# P1 = -R⁻¹ @ t
# 第四步：用 Rᵀ = R⁻¹
# P1 = -Rᵀ @ t
# 这个 P1 就是：相机 2 的原点，在相机 1 坐标系中的位置。
# 所以：C2_in_1 = -R.T @ t

#计算第二台相机在第一台相机坐标中的位置
C2_in_1=-R.T @ t  # 3x3 @ 3x1 -> 3x1

print("第二台相机在第一台相机坐标中的位置:")
print(C2_in_1)
print("C2_in_1 shape =",C2_in_1.shape) #3,1


#组成两帧的简化轨迹
#第一台相机放在原点
#第二台相机的位置用c2_in_1表示
trajectory_two_frames=np.vstack( #np.vstack：竖直堆叠
    [
        np.zeros((1,3)), #相机 1 的位置，放在原点 (0, 0, 0)
        C2_in_1.reshape(1,3) #相机 2 的位置，形状从 (3, 1) 变成 (1, 3)
    ]
)

print("两帧轨迹 shape=",trajectory_two_frames.shape) #2,3
print("两帧轨迹:")
print(trajectory_two_frames)








#第 10 步：准备 VO 的多帧输入 从视频中每隔几帧提取一张图片，得到连续的图像序列
# 为什么不能每一帧都处理？
# 假设视频是 30 FPS，拍 10 秒就是大约 300 帧。
# 如果每一帧都做：ORB 特征
# → 特征匹配
# → RANSAC
# → 相机运动
# 计算量会很大，而且相邻帧变化太小。
# 所以先每隔 15 帧取一帧：
# 第 0 帧
# 第 15 帧
# 第 30 帧
# 第 45 帧
# ...
# 这样既保留连续运动，又减少计算量。
# 新建抽帧脚本 在extract_frames.py中
