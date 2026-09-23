#第一步：用 Open3D 把 .xyz 文件读进来并打印信息
from pathlib import Path
import open3d as o3d
import numpy as np

# 找到当前脚本旁边的results/point_c.xyz
xyz_path=Path(__file__).with_name("results") / "points_c.xyz"

# #Open3D 读取点云
# pcd=o3d.io.read_point_cloud(str(xyz_path))

# 使用 NumPy 读取文本点云，避免 Open3D 直接读取中文路径
points = np.loadtxt(xyz_path)

# 创建空的 Open3D 点云对象
pcd = o3d.geometry.PointCloud()

# 把 NumPy 点云交给 Open3D
pcd.points = o3d.utility.Vector3dVector(points)

print("点云文件:",xyz_path)
print("点云对象:",pcd)
print("点数:",len(pcd.points))
print("是否有颜色:",pcd.has_colors())




#第二步：加上颜色和坐标轴，并打开真正的三维窗口
#给点云统一染成红色，rgb范围是0-1
pcd.paint_uniform_color([1.0,0.0,0.0])  

#创建一个坐标轴
#红色：x轴
#绿色：y轴
#蓝色：z轴
axis=o3d.geometry.TriangleMesh.create_coordinate_frame(  #创建坐标轴
    size=0.1,
    origin=[0.0,0.0,0.0]    #坐标轴起点放在相机原点
)

#打开可视化窗口
o3d.visualization.draw_geometries(  #打开交互式三维窗口
    [pcd,axis],
    window_name="3.3 depth to point cloud"
)

# # 计算点云中心
# center = points.mean(axis=0)

# # 让相机看向点云中心
# o3d.visualization.draw_geometries(
#     [pcd, axis],
#     window_name="3.3 depth to point cloud",
#     lookat=center,
#     zoom=0.5,
# )

