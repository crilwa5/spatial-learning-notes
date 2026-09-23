#RGB 图像 + 深度图 + 相机内参 → 彩色点云

#第一步：人工生成一张 20×20 的彩色图像
import numpy as np
import open3d as o3d
from pathlib import Path
import copy

#和深度图保持一样的大小
height=20
width=20

#图像上的行，列坐标
rows,cols=np.mgrid[0:height,0:width].astype(float)

#红色通道：从左到右逐渐增强
# 拆开看：cols：列号，从 0 到 19。width - 1 = 19。
# cols / 19：把列号归一化到 0 到 1 之间。
# × 255：放大到 0 到 255，这是像素值的范围。
# .astype(np.uint8)：转成 8 位无符号整数（0–255）。
# 效果：最左边 cols = 0 → 红色 = 0。
# 最右边 cols = 19 → 红色 = 255。
# 所以红色通道从左到右逐渐增强
red=(cols/(width-1)*255).astype(np.uint8) #20x20

#绿色通道：从上到下逐渐增强
# rows：行号，从 0 到 19。
# rows / 19：归一化到 0 到 1。
# × 255：放大到 0 到 255。
# 效果：最上面 rows = 0 → 绿色 = 0。
# 最下面 rows = 19 → 绿色 = 255。
# 所以绿色通道从上到下逐渐增强
green=(rows/(height-1)*255).astype(np.uint8)  #20x20

#蓝色通道：先统一设为一个中间值
# np.full((20, 20), 128)：生成一个 20×20 的数组，每个值都是 128。
# dtype=np.uint8：数据类型是 8 位无符号整数。
blue=np.full((height,width),128,dtype=np.uint8) #20x20

#把三个通道叠成(H,W,3)的彩色图像
# np.stack 的作用:np.stack 会把几个数组沿着一个新维度叠起来。
# 这里：输入是三个 (20, 20) 的数组。
# axis=-1 表示在最后一个维度上叠。
# 结果形状是 (20, 20, 3)。
# 用一个小例子理解
# 假设：
# red   = [[1, 2],
#          [3, 4]]
# green = [[5, 6],
#          [7, 8]]
# blue  = [[9, 10],
#          [11, 12]]
# 调用：np.stack([red, green, blue], axis=-1)
# 结果：
# [
#   [[1, 5, 9],   [2, 6, 10]],
#   [[3, 7, 11],  [4, 8, 12]]
# ]
# 形状是 (2, 2, 3)。
color_image=np.stack([red,green,blue],axis=-1)  
#np.stack：把三个数组叠在一起。
#axis=-1：在最后一个维度上叠。

print("color_image.shape =", color_image.shape) #20,20,3
print("color_image.dtype =", color_image.dtype) #uint8
print("color_image.min() =", color_image.min())
print("color_image.max() =", color_image.max())
print("左上角颜色 =", color_image[0, 0])
print("右下角颜色 =", color_image[-1, -1])



# 和彩色图像使用同一个主点
cx = 9.5
cy = 9.5

# 构造与颜色图像对应的深度图
# 中心深度小，边缘深度大
depth_map = 1.2 + 0.5 * (
    ((cols - cx) / 10.0) ** 2
    + ((rows - cy) / 10.0) ** 2
)

# 相机内参
K = np.array(
    [
        [50.0, 0.0, cx],
        [0.0, 50.0, cy],
        [0.0, 0.0, 1.0],
    ]
)

print("depth_map.shape =", depth_map.shape)#20x20
print("depth_map.min() =", depth_map.min())
print("depth_map.max() =", depth_map.max())
print("K.shape =", K.shape)#3x3

















#第二步：逐像素同时生成三维点和颜色
# 核心对应关系是：
# 第 (row, col) 个像素的深度
#   → 一个三维点
# 第 (row, col) 个像素的 RGB
#   → 这个三维点的颜色
#分别收集三维点和颜色
points=[]
colors=[]

#从内参中取出焦距
fx=K[0,0]
fy=K[1,1]

#遍历每一个像素
for row in range(height):
    for col in range(width):
        depth=float(depth_map[row,col])
        if depth<=0:
            continue

        u=float(col)
        v=float(row)

        x_c=(u-cx)/fx
        y_c=(v-cy)/fy
        ray_c=np.array([x_c,y_c,1.0])

        point_c=ray_c*depth
        points.append(point_c)

        #取出当前rgb，并归一化到0-1
        rgb=color_image[row,col].astype(float)/255.0
        colors.append(rgb)

points=np.array(points)
colors=np.array(colors)

print("points.shape =", points.shape)#400,3
print("colors.shape =", colors.shape)#400,3
print("colors.min() =", colors.min())
print("colors.max() =", colors.max())
















#第三步，点云对象  可视化 窗口
# 创建 Open3D 点云对象
pcd = o3d.geometry.PointCloud()

# 设置三维坐标
pcd.points = o3d.utility.Vector3dVector(points)

# 设置每个点的颜色
pcd.colors = o3d.utility.Vector3dVector(colors)

print("Open3D 点数 =", len(pcd.points))#400
print("是否有颜色 =", pcd.has_colors())#true
print("点云中心 =", pcd.get_center())


# 创建坐标轴
axis = o3d.geometry.TriangleMesh.create_coordinate_frame(
    size=0.2,
    origin=[0.0, 0.0, 0.0],
)

# 让相机看向点云中心
center = points.mean(axis=0)

# 打开彩色点云窗口
o3d.visualization.draw_geometries(
    [pcd, axis],
    window_name="RGB-D colored point cloud",
    width=1280,
    height=720,
    lookat=center,
    zoom=0.5,
)















 
#第四步：保存彩色点云   把点云和颜色拼在一起，保存成一个带颜色的点云文件
#前3列：x，y，z
#后3列：r,g,b 范围转换为0-255
output_dir=Path(__file__).with_name("results")
output_dir.mkdir(exist_ok=True)

#把颜色转成 0–255 之前归一化了
#np.round(...)：四舍五入成整数
rgb255=np.round(colors * 255.0)

#把点和颜色拼在一起
# np.column_stack：按列拼接。
# 效果：把两个 (N, 3) 数组合并成一个 (N, 6) 数组。
# 结果：points_rgb =
# [[x1, y1, z1, r1, g1, b1],
#  [x2, y2, z2, r2, g2, b2],
#  ...
#  [xN, yN, zN, rN, gN, bN]]
points_rgb=np.column_stack([points,rgb255])

colored_path=output_dir/"colored_points.xyzrgb"
np.savetxt(
    colored_path,
    points_rgb,
    #每一列的格式
    fmt=["%.6f", "%.6f", "%.6f", "%.0f", "%.0f", "%.0f"]#后三个是整数
)

print("已保存彩色点云:",colored_path)


















#第五步：体素降采样实验
# 体素降采样可以理解成：
# 把三维空间划分成很多小立方体
# 每个小立方体只保留一个代表点
# 这样点数会减少，点云更稀疏，但整体形状仍然保留
#使用不同大小的体素做降采样
voxel_sizes=[0.02,0.05,0.1,0.2]
# 0.02：体素很小，保留的点比较多；
# 0.05：体素变大，点数进一步减少；
# 0.1：形状更粗；
# 0.2：体素很大，点数会明显减少，细节可能丢失

for voxel_size in voxel_sizes:
    down_pcd=pcd.voxel_down_sample(voxel_size=voxel_size)

    print(
        "voxel_size =",
        voxel_size,
        "降采样后点数 =",
        len(down_pcd.points)  #400,329,137,50
    )
    # 1. 0.02 的体素比点之间的距离还小，所以每个点基本都保留；
    # 2. 体素越大，多个点被合并成一个代表点；
    # 3. 体素太大时，形状会变粗，细节会丢失


#原始点云 vs 体素降采样点云
#选择一个中间大小的体素
down_pcd_01=pcd.voxel_down_sample(voxel_size=0.1)

#复制点云，只用于把两个点云左右摆开位置
pcd_display=copy.deepcopy(pcd)
down_display=copy.deepcopy(down_pcd_01)

#原始点云向左移动
pcd_display.translate((-0.5,0.0,0.0))

#降采样点云向右移动
down_display.translate((0.5,0.0,0.0))

#把两个点云和坐标轴一起显示
o3d.visualization.draw_geometries(
    [pcd_display,down_display,axis],
    window_name="original vs voxel downsampled",
    width=1280,
    height=720,
    lookat=[0.0,0.0,center[2]]
)















#第六步：从点云创建真正的体素网格
voxel_grid=o3d.geometry.VoxelGrid.create_from_point_cloud(
    pcd,
    voxel_size=0.1
)

print("体素数量 =",len(voxel_grid.get_voxels()))  #137
# 原来的 400 个点
#   → 被划分到很多 0.1 × 0.1 × 0.1 的小立方体里
#   → 只有 137 个小立方体里面有点
#   → 这 137 个被占用的小立方体组成了 VoxelGrid
#上面对比那种是留点，现在这种是格子（体素）

#显示体素网格和坐标轴
o3d.visualization.draw_geometries(
    [voxel_grid,axis],
    window_name="voxel grid",
    width=1280,
    height=720,
    lookat=[0.0,0.0,center[2]]
)










#第七步：点云法向量估计
#法向量可以理解为：每个点所在表面的朝向
#估计点云法向量
pcd.estimate_normals(#对点云中每个点，估计它所在表面的法向量方向  估计出来的
    search_param=o3d.geometry.KDTreeSearchParamHybrid(
        radius=0.1,#在半径 0.1 的邻域内寻找附近点，估计表面方向
        max_nn=30  #最多使用最近 30 个邻居
    )
    # 核心思想：
    # 对一个点 P：找到它附近的邻居点。用这些邻居点拟合一个局部平面。这个平面的垂直方向，就是法向量
)

print("是否有法向量 =",pcd.has_normals())
print("法向量 shape =",np.asarray(pcd.normals).shape)  #400，3   3是坐标x，y，z



#显示点云和法向量
o3d.visualization.draw_geometries(
    [pcd,axis],
    window_name="point normals",
    width=1280,
    height=720,
    point_show_normal=True, #显示每个点附近的法向量方向
    lookat=[0.0,0.0,center[2]]  #相机看向的位置
)











#第八步：从结构化点云生成网格
#点云只有点和颜色，网格需要额外定义“哪些点连成三角形”。
#因为我们的点云本来就是从 20 × 20 深度图逐行逐列生成的，所以可以直接按网格连接点，不需要复杂的重建算法
#什么意思？
# 现在我们有 400 个点（来自 20×20 的深度图），想把它变成网格。
# 怎么变？关键：这 400 个点不是乱排的，它们是从 20×20 的深度图逐行逐列生成的。
# 也就是说：第 0 行有 20 个点。第 1 行有 20 个点。……第 19 行有 20 个点。
# 总共 20×20 = 400 个点。
# 它们的排列是规则的，像一张方格纸。
# 所以我们不需要复杂的重建算法，直接按格子连接就行



#1.把点（400,3）恢复成（20,20,3）
vertices=points.reshape(height,width,3)

#2.每个小方格拆成两个三角形也就是每个2x2的像素块生成2个三角形
# 想象 20×20 的网格，相邻的 4 个点组成一个小方格：
# i0 ─── i1
# │      │
# i2 ─── i3
# i0：左上角   i1：右上角    i2：左下角    i3：右下角
# 怎么给它们编号？是这400个即0-399中的哪个
# i0 = row * width + col
# i1 = i0 + 1
# i2 = i0 + width
# i3 = i2 + 1
# 解释：
# row * width + col：第 row 行、第 col 列的点，在 400 个点中的编号。
# i0 + 1：右边那个点。
# i0 + width：下面那个点。
# i2 + 1：右下角那个点。
# 一个方格有 4 个顶点，可以拆成两个三角形：
# 三角形 1：i0, i1, i3
# 三角形 2：i0, i3, i2
# 画出来：
# i0 ─── i1
# │  ╲   │
# │   ╲  │
# i2 ─── i3

faces=[]


#要拆小格子，需要相邻的两列or两行
# 看行方向  例如3x3  
# row = 0：用第 0 行和第 1 行 → 可以组成方格。
# row = 1：用第 1 行和第 2 行 → 可以组成方格。
# row = 2：用第 2 行和第 3 行 → 但第 3 行不存在！
# 所以 row 最多只能到 1，也就是 height - 2 = 1。所以height-1，在range中height-1不取，取到-2，列同理
# 因为一个小方格需要 4 个点，也就是相邻的两行、两列。
# 最后一行和最后一列没有“下一行/下一列”可以配对，所以循环只能到 height - 1 和 width - 1。
for row in range(height-1):
    for col in range(width-1):
        #四个相邻顶点的编号
        i0=row * width +col
        i1=i0+1
        i2=i0+width
        i3=i2+1

        #每个方形格子拆成2个三角形
        faces.append([i0,i1,i3])
        faces.append([i0,i3,i2])

faces=np.array(faces,dtype=np.int32)

# 20 行，能组成方格的行数：20 - 1 = 19
# 20 列，能组成方格的列数：20 - 1 = 19
# 所以：
# 19 × 19 = 361 个小方格
# 361 × 2 = 722 个三角形


#创建三角网格
mesh=o3d.geometry.TriangleMesh()   #创建一个空的三角网格
mesh.vertices=o3d.utility.Vector3dVector(points)  #顶点，就是 400 个点
mesh.triangles=o3d.utility.Vector3iVector(faces)  #三角面，就是 722 个三角形

#把颜色也放到网格顶点上
mesh.vertex_colors=o3d.utility.Vector3dVector(colors)  

#计算网格法向量，便于显示光照  显示网格时，能看到明暗变化，看起来有立体感
mesh.compute_vertex_normals()

print("顶点数量 =",len(mesh.vertices))  #20x20=400
print("三角面数量 =",len(mesh.triangles)) #前面算的，722个

#显示网格
o3d.visualization.draw_geometries(
    #mesh：网格
    [mesh,axis],
    window_name="structured mesh",
    width=1280,
    height=720,
    lookat=[0.0,0.0,center[2]]
)

# 保存网格顶点和三角面
mesh_data_path = output_dir / "structured_mesh.npz"

np.savez(
    mesh_data_path,
    vertices=points,
    faces=faces,
)

print("已保存网格数据:", mesh_data_path)