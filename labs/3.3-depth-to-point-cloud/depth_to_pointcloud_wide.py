#生成一个更宽的深度图，让点云看起来像一个弯曲的表面，而不是一条很细的线段
#第一步：只生成深度图  NumPy 生成一个 20×20 的深度图
import numpy as np
from pathlib import Path

#深度图尺寸 20行 20列
height=20
width=20

#主点放在图像中心附近
cx=9.5
cy=9.5

#生成每一行，每一列的坐标网格
#rows.shape=(height,width)
#cols.shape=(height,width)

# 生成两个二维网格数组：rows 和 cols。
# 每个数组的形状都是 (height, width)。
# 把数据类型转成浮点数。

# np.mgrid 是 NumPy 里的一个函数，用来生成网格坐标。
# 它的写法：np.mgrid[0:height, 0:width]
# 意思是：第一个维度：从 0 到 height（不含 height）。
# 第二个维度：从 0 到 width（不含 width）。
# 它返回两个数组：第一个数组：每一行都是行号。
# 第二个数组：每一列都是列号。

# height = 3     width = 4
# 调用：rows, cols = np.mgrid[0:3, 0:4]
# 结果：
# rows =
# [[0 0 0 0]
#  [1 1 1 1]
#  [2 2 2 2]]
# cols =
# [[0 1 2 3]
#  [0 1 2 3]
#  [0 1 2 3]]
# 解释：rows：每个位置存的是行号。cols：每个位置存的是列号。
# 比如位置 (1, 2)：rows[1, 2] = 1    cols[1, 2] = 2
# 表示这个位置是第 1 行、第 2 列。

# 为什么要生成这两个数组？
# 因为深度图转点云时，需要知道每个像素的坐标 (u, v)。
# u 就是列号（横向位置）。 v 就是行号（纵向位置）。
# 有了 rows 和 cols，就可以对每个像素：
# u = cols[v, u]列号所以cols       v = rows[v, u]行号所以rows
# 然后结合深度 Z 和内参，算出三维点

# np.mgrid 默认生成的是整数
# .astype(float) 把数组转成浮点数
rows,cols=np.mgrid[0:height,0:width].astype(float)  #两个数组都是20x20


#构造一个弯曲的深度图
#越靠中心，深度越小，越靠边缘，深度越大
#因为这样看起来像一个弯曲的碗或球面，中间离相机近，边缘离相机远
depth_map=1.2+0.5*(
    ((cols-cx)/10.0)**2
    +((rows-cy)/10.0)**2
)

#相机内参
K=np.array(
    [
        [50.0, 0.0, cx],
        [0.0, 50.0, cy],
        [0.0, 0.0, 1.0],
    ]
)

print("depth_map.shape =", depth_map.shape)#20,20
print("depth_map.min() =", depth_map.min())
print("depth_map.max() =", depth_map.max())
print("K.shape =", K.shape) #3x3



#第二步：把整张宽深度图转成点云
#用一个列表收集所有三维点
points=[]

#从内参中取出焦距
fx=K[0,0]
fy=K[1,1]

#遍历深度图的每一行每一列
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

points=np.array(points)

print("points.shape =", points.shape)  #400,3
print("点数 =", len(points)) #400
print("x 范围 =", points[:, 0].min(), "到", points[:, 0].max())
print("y 范围 =", points[:, 1].min(), "到", points[:, 1].max())
print("z 范围 =", points[:, 2].min(), "到", points[:, 2].max())






#第三步：保存点云
#结果保存在results文件夹中
output_dir=Path(__file__).with_name("results")
output_dir.mkdir(exist_ok=True)

#保存numpy点云
npy_path=output_dir / "points_wide.npy"
np.save(npy_path,points)

#保存普通文本点云
xyz_path=output_dir/"points_wide.xyz"
np.savetxt(
    xyz_path,
    points,
    fmt="%.6f"
)

print("已保存 npy:", npy_path)
print("已保存 xyz:", xyz_path)







