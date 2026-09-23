# 3.3 Depth and RGB-D Point Cloud

## 实验目标

把深度图转换成三维点云，并为点云加入颜色、体素、法向量和网格。

## 核心关系

深度图：
(H, W)

像素 + 深度：
P_c = depth * ray_c

彩色点云：
points: (N, 3)
colors: (N, 3)，范围为 0～1

体素降采样：
多个点落在同一个体素时，只保留一个代表点。

VoxelGrid：
每个被占用的体素显示为一个三维小立方体。

网格：
顶点 + 三角面。

## 实验文件

- depth_to_pointcloud.py
- depth_to_pointcloud_wide.py
- rgbd_to_pointcloud.py
- visualize_points.py
- visualize_points_wide.py

## 运行

python labs/3.3-depth-to-point-cloud/depth_to_pointcloud.py
python labs/3.3-depth-to-point-cloud/rgbd_to_pointcloud.py

## 实际结果

4×5 深度图：
有效点数：19
points.shape: (19, 3)

20×20 深度图：
点数：400
points.shape: (400, 3)
colors.shape: (400, 3)

体素降采样：
voxel_size = 0.02，400 点
voxel_size = 0.05，329 点
voxel_size = 0.10，137 点
voxel_size = 0.20，50 点

网格：
顶点数：400
三角面数：722

## 遇到的问题

1. Open3D 读取中文路径下的 .xyz 失败
   - 解决：使用 NumPy 读取文件，再交给 Open3D 创建 PointCloud

2. Open3D 显示窗口出现 SetViewPoint 警告
   - 解决：不影响显示，后续可调整窗口参数

3. 点云看起来很小
   - 解决：原来的合成场景范围太窄，后来制作了更宽的 20×20 深度图