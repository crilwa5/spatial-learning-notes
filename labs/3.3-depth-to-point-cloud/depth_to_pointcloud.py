import numpy as np
from pathlib import Path

#深度图：高度4，宽度5
#每个数字表示对应像素的深度
#0表示这个像素没有有效深度
depth_map=np.array([
    [1.0, 1.0, 1.0, 1.0, 1.0],
    [1.0, 2.0, 0.0, 2.0, 1.0],
    [1.0, 2.0, 3.0, 2.0, 1.0],
    [1.0, 1.0, 1.0, 1.0, 1.0],
],
dtype=float
)

#相机内参
#fx，fy：焦距
#cx，cy：主点
K=np.array(
    [
        [100.0, 0.0, 2.0],
        [0.0, 100.0, 1.5],
        [0.0, 0.0, 1.0]
    ]
)






#第一步：观察深度图和内参的形状
height,width=depth_map.shape   

print("depth_map.shape =", depth_map.shape)#4,5
print("height =", height) #4
print("width =", width) #5
print("K.shape =", K.shape) #3x3







#第二步：从深度图中取出一个像素和它的深度
# 这里要特别分清数组索引和图像坐标：
# depth_map[row, col]
# row = 高度方向的位置
# col = 宽度方向的位置
# v = row         u = col
# 也就是说：u：图像的横向位置
# v：图像的纵向位置
#选取中间位置：第二行第二列
row=2
col=2

#图像坐标：u对应列，v对应行
u=float(col)
v=float(row)

#取出这个像素对应的深度
depth=float(depth_map[row,col])

print("row =", row)
print("col =", col)
print("u =", u)
print("v =", v)
print("depth =", depth)







#第三步：用这个像素生成相机坐标射线和三维点
#从内参中取出相机参数
fx = K[0, 0]
fy = K[1, 1]
cx = K[0, 2]
cy = K[1, 2]

# 像素 -> 相机坐标射线方向
x_c=(u-cx)/fx
y_c=(v-cy)/fy

ray_c=np.array([x_c,y_c,1.0])

#射线+深度->相机坐标三维点
point_c=ray_c*depth

print("ray_c =", ray_c)
print("point_c =", point_c)







#第四步：遍历整张深度图，把所有有效像素变成三维点
#用一个列表收集所有三维点
points=[]

#遍历深度图的每一行，每一列
for row in range(height):
    for col in range(width):
        #取当前像素的深度
        depth=float(depth_map[row][col])

        #深度为0表示无效像素，跳过
        if depth<=0:
            continue

        #当前像素的射线
        u=float(col)
        v=float(row)

        x_c=(u-cx)/fx
        y_c=(v-cy)/fy
        ray_c=np.array([x_c,y_c,1.0])

        #射线+深度->相机坐标三维点
        point_c=ray_c*depth

        #加入列表
        points.append(point_c)

#把列表转成numpy数组，形状为(N,3)
points=np.array(points)

print("有效点数量:",len(points))
print("points.shape =",points.shape)
print("前5个点:")
print(points[:5])







#第五步：观察点云的坐标范围
# 点云不是一个抽象数组，而是每个点都对应空间位置
# points[:, 0] = x
# points[:, 1] = y
# points[:, 2] = z
#分别取出x，y，z三个坐标
x_values=points[:,0]
y_values=points[:,1]
z_values=points[:,2]

print("x 范围:",x_values.min(),"到",x_values.max())
print("y 范围:",y_values.min(),"到",y_values.max())
print("z 范围:",z_values.min(),"到",z_values.max())

#看看深度图里一共有哪些深度
# np.unique 是 NumPy 里的一个函数，作用是：对一个数组，找出其中所有唯一值（去重），并按从小到大排序。
# 默认情况下，np.unique 只返回唯一值。
# 但如果加上：return_counts=True，它还会额外返回每个唯一值出现了多少次。
# 它做了两件事：
# 找出 z_values 里所有不同的值，去掉重复。
# 统计每个不同的值出现了多少次
unique_z,counts=np.unique(z_values,return_counts=True)

print("不同深度值:",unique_z)
print("每个深度对应的点数:",counts)








#第六步：把点云保存下来
# 要保存两种格式：
# points_c.npy：给 Python 读取，保留 NumPy 数组格式
# points_c.xyz：普通文本点云，适合 Open3D、CloudCompare 等工具查看
#结果保存在当前脚本旁边的results文件夹中
# Path(__file__) 表示当前 Python 文件的位置；
# .with_name("results") 表示在脚本同目录下创建一个 results 文件夹；
# .mkdir(exist_ok=True) 表示文件夹已经存在时不报错
output_dir=Path(__file__).with_name("results")
output_dir.mkdir(exist_ok=True)

#保存numpy点云
npy_path=output_dir / "points_c.npy"
#np.save() 保存 Python/NumPy 使用的二进制点云；
np.save(npy_path,points)

#保存普通文本点云，每行是x，y，z
xyz_path=output_dir / "points_c.xyz"
#np.savetxt() 保存人类也能直接查看的文本点云。
np.savetxt(
    xyz_path,
    points,
    fmt="%.6f"
)

print("已保存 npy:",npy_path)
print("已保存 xyz:",xyz_path)







#第七步：把保存的点云重新读回来，检查保存前后是否一致
# 重新读取 NumPy 点云
loaded_points_npy=np.load(npy_path)

# 重新读取普通文本点云
loaded_points_xyz=np.loadtxt(xyz_path)

print("原始points.shape =",points.shape)
print("读取 npy.shape =", loaded_points_npy.shape)
print("读取 xyz.shape =", loaded_points_xyz.shape)

# np.allclose 用来判断两个数组是否在数值上近似相等
# np.allclose 是 NumPy 里的一个函数，作用是：判断两个数组是否在数值上近似相等。
# 它的判断标准不是“完全一样”，而是：每个对应位置的差值，是否都在一个很小的容忍范围内。
# 如果所有元素都足够接近，返回 True。只要有一个元素差太多，返回 False。
print("npy 和原数组一致:",np.allclose(loaded_points_npy,points))
print("xyz 和原数组一致:",np.allclose(loaded_points_xyz,points))

