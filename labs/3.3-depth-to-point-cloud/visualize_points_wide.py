from pathlib import Path

import numpy as np
import open3d as o3d

# 找到当前脚本旁边的 results/points_wide.xyz
xyz_path=Path(__file__).with_name("results")/"points_wide.xyz"

#使用numpy读取文本点云，避免open3d读取中文路径
points=np.loadtxt(xyz_path)

#创建open3d点云对象
pcd=o3d.geometry.PointCloud()
pcd.points=o3d.utility.Vector3dVector(points)

print("点云文件:",xyz_path)
print("numpy点云 shape:",points.shape)  #400,3
print("Open3D 点数:",len(pcd.points))#400
print("是否有颜色:",pcd.has_colors())


#给点云染成蓝色
pcd.paint_uniform_color([0.1,0.5,1.0])

#创建坐标轴
#红色：x轴
#绿色：y轴
#蓝色：z轴
axis=o3d.geometry.TriangleMesh.create_coordinate_frame(
    size=0.2,
    origin=[0.0,0.0,0.0]
)

#计算点云中心，让相机看向点云
center=points.mean(axis=0)

#打开三维窗口
o3d.visualization.draw_geometries(
    [pcd,axis],
    window_name="3.3 wide point cloud",
    lookat=center,
    zoom=0.5
)
