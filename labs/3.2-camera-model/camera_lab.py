#给定一个世界坐标中的三维点，计算它在相机中的坐标，并最终投影到像素坐标。
import numpy as np

#R_cw:世界坐标->相机坐标的旋转矩阵
#单位矩阵表示相机坐标轴暂时和世界坐标轴方向一致
#相机坐标轴和世界坐标轴方向暂时一致，也就是相机没有旋转
R_cw=np.eye(3)  #生成一个 3×3 的单位矩阵
# [1  0  0]
# [0  1  0]
# [0  0  1]



#相机中心在世界坐标中的位置
camera_center_w=np.array([1.0,0.0,0.0])



#t_cw:外参平移
#关系是：P_c=R_cw @ P_w + t_cw
#相机坐标的原点是相机中心,所以要让相机坐标p_camera=0,0,0
#而p_camera=p_cw @ camera_center_w + t_cw  p_cw=E 
#所以0,0,0=1,0,0 +  x,y,z
#x=-1,y=0,z=0
#所以t_cw=(-1,0,0) 即-camera_center_w

#other  
# 要把世界坐标的点转到相机坐标，需要：
# 先把世界坐标原点移到相机中心：减去 camera_center_w。
# 再旋转：乘 R_cw。
# 所以公式是：
# P_c = R_cw @ (P_w - camera_center_w)
#     = R_cw @ P_w - R_cw @ camera_center_w
# 其中R_cw=E   所以t_cw=-camera_center_w
#相对位置 = 物体位置 − 你的位置
#比如你站在 (1, 0, 0)，杯子在 (3, 0, 0)。
#杯子相对于你是在 3,0,0 - 1,0,0 = 2,0,0
t_cw=-camera_center_w



#K:相机内参矩阵
#fx，fy 是焦距，cx，cy是主点
K=np.array([
    [800.0,0.0,640.0],
    [0.0,800.0,360.0],
    [0.0,0.0,1.0]
])
# fx = 800：水平方向焦距。
# fy = 800：垂直方向焦距。
# cx = 640：主点横向位置。
# cy = 360：主点纵向位置。


#第二步：世界坐标点转成相机坐标点

#一个世界坐标中的三维点
point_w=np.array([1.3,-0.2,2.0])

#先观察形状和值，不做任何计算
print("R_cw.shape",R_cw.shape)  #3，3
print("camera_center_w.shape",camera_center_w.shape)  #3，
print("t_cw =",t_cw)   #3，
print("K.shape",K.shape)   #3,3
print("point_w.shape =",point_w.shape)  #3，
print("point_w =",point_w) 

#世界坐标转成相机坐标
point_c=R_cw @ point_w + t_cw
print("point_c.shape =", point_c.shape)  #3，
print("point_c =", point_c)  #0.3，-0.2,2.0








#第三步：相机坐标点投影成像素坐标  p = K @ P_c
# K @ P_c 得到的是一个三维齐次像素向量，最后还要除以第三个分量，才能得到真正的二维像素坐标：
#K @ P_c 得到的是齐次像素向量，第三个分量是深度 Z。
#要得到真正的像素坐标，必须除以第三个分量，这一步叫透视除法，实现“近大远小”。
# u = 第一个分量 / 第三个分量          v = 第二个分量 / 第三个分量
#why？
#像素公式：u = fx × (X / Z) + cx    v = fy × (Y / Z) + cy
# K = [ fx   0   cx ]
#     [  0  fy   cy ]
#     [  0   0    1 ]
# 做矩阵乘法：
# K @ P_c = [ fx × X + cx × Z ]
#           [ fy × Y + cy × Z ]
#           [        Z        ]
#第三个分量是 Z，也就是深度，所以每一项还得除以Z
#才能取到u = fx × (X / Z) + cx    v = fy × (Y / Z) + cy

# 相机坐标 -> 像素坐标
# K @ point_c 得到齐次像素向量
projected=K @ point_c

print("projected.shape =", projected.shape) #3,
print("projected =", projected)

# 透视除法：除以第三个分量
u=projected[0]/projected[2]
v=projected[1]/projected[2]
pixel=np.array([u,v])

print("pixel.shape =", pixel.shape) #3,
print("pixel =", pixel)









#第四步：像素坐标反投影成相机坐标射线
# 注意：像素不能恢复一个确定的三维点，只能恢复一条射线。
# 公式可以从 K 反推：
# x_c = (u - cx) / fx (*z)
# y_c = (v - cy) / fy (*z)
# z_c = 1 (*z)
# z省略，因为不确定深度，所以都除z，相当于没有（）内的那个
# 这里的 z_c = 1 不是说深度一定是 1，而是我们先取一个统一的射线尺度。真正的位置还要乘上深度。

# 像素坐标 -> 相机坐标射线
# 从 K 中取出内参
fx=K[0][0]
fy=K[1][1]
cx=K[0][2]
cy=K[1][2]

#射线方向：z_c=1只是取一个统一尺度
ray_c=np.array([
    (pixel[0]-cx)/fx,
    (pixel[1]-cy)/fy,
    1.0
])

print("ray_c.shape",ray_c.shape) #3,
print("ray_c =",ray_c)









#第五步：给射线一个深度，恢复相机坐标点  P_c = ray_c × depth
# 给射线一个已知深度
depth = 2.0

# 用射线和深度恢复相机坐标点
recovered_c=ray_c * depth

print("depth =",depth) #2.0
print("recovered_c.shape",recovered_c.shape) #3,
print("recovered_c =",recovered_c)








#第六步：把恢复出的相机坐标点转回世界坐标
# 原来的公式是：P_c = R_cw @ P_w + t_cw
# R_cw的逆=其转置
# 反过来就是：P_w = R_cw.T @ (P_c - t_cw)
# 这里的 R_cw.T 表示旋转矩阵的转置。单位矩阵的转置还是自己

# 相机坐标 -> 世界坐标
recovered_w= R_cw.T @ (recovered_c-t_cw)

print("recovered_w.shape =", recovered_w.shape) #3
print("recovered_w =", recovered_w)  
print("original point_w =", point_w)






#第七步：把恢复出的世界点重新投影成像素，计算重投影误差
#把恢复出的世界点重新投影
reprojected_c=R_cw @ recovered_w + t_cw
reprojected_h=K @ reprojected_c

#透视除法，得到重新投影的像素
reprojected_pixel=np.array([
    reprojected_h[0]/reprojected_h[2],
    reprojected_h[1]/reprojected_h[2]
])

#重投影误差：新旧像素之间的欧氏距离
# 对任意维度，两个点 A 和 B 的欧氏距离是：
# 距离 = √( (A₁-B₁)² + (A₂-B₂)² + ... + (An-Bn)² )
# 也就是：每个维度的差值的平方，加起来，再开根号。
# reprojected_pixel - pixel：两个像素坐标的差值，是一个二维向量。
# np.linalg.norm(...)：求这个向量的长度，也就是欧氏距离
reprojection_error=np.linalg.norm(reprojected_pixel-pixel)

print("reprojected_pixel =", reprojected_pixel)
print("original_pixel =", pixel)
print("reprojection_error =", reprojection_error)



#封装函数
#1.世界坐标转相机坐标
def world_to_camera(point_w,R_cw,t_cw):
    """
    把世界坐标中的三维点转换到相机坐标。

    参数：
        point_w: 世界坐标点，形状 (3,)
        R_cw:    世界坐标到相机坐标的旋转矩阵，形状 (3, 3)
        t_cw:    外参平移向量，形状 (3,)

    返回：
        point_c: 相机坐标点，形状 (3,)
    """
    return R_cw @ point_w + t_cw


# 用函数重新计算一次，检查结果是否和前面一致
point_c_from_function = world_to_camera(point_w, R_cw, t_cw)
print("point_c_from_function =", point_c_from_function)


#2.相机坐标转像素坐标
def project_camera_to_pixel(point_c, K):
    """
    把相机坐标中的三维点投影成像素坐标。
    参数：
        point_c: 相机坐标点，形状 (3,)
        K:       相机内参矩阵，形状 (3, 3)
    返回：
        pixel: 像素坐标，形状 (2,)
    """
    # 得到齐次像素向量
    projected_h = K @ point_c

    # 透视除法：除以第三个分量
    u = projected_h[0] / projected_h[2]
    v = projected_h[1] / projected_h[2]

    return np.array([u, v])

# 使用函数重新计算，检查结果是否和之前一致
pixel_from_function = project_camera_to_pixel(point_c_from_function, K)
print("pixel_from_function =", pixel_from_function)


#3.像素坐标转相机坐标射线
def pixel_to_camera_ray(pixel,K):
    """
    把像素坐标反投影成相机坐标中的射线方向。

    参数：
        pixel: 像素坐标，形状 (2,)
        K:     相机内参，形状 (3, 3)

    返回：
        ray_c: 相机坐标射线，形状 (3,)
               ray_c[2] = 1，表示使用统一尺度
    """
    fx=K[0][0]
    fy=K[1][1]
    cx=K[0][2]
    cy=K[1][2]

    x_c=(pixel[0]-cx)/fx
    y_c=(pixel[1]-cy)/fy

    return np.array([x_c,y_c,1.0])

# 使用函数重新计算一次
ray_c_from_function = pixel_to_camera_ray(pixel_from_function, K)
print("ray_c_from_function =", ray_c_from_function)




#4.射线 + 深度恢复相机坐标点
def point_on_camera_ray(ray_c,depth):
    """
    根据相机坐标射线和深度恢复相机坐标点。

    参数：
        ray_c: 相机坐标射线，形状 (3,)
               约定 ray_c[2] = 1
        depth: 沿 z 轴的深度

    返回：
        point_c: 相机坐标点，形状 (3,)
    """
    return ray_c * depth

# 使用函数恢复相机坐标点
recovered_c_from_function = point_on_camera_ray(
    ray_c_from_function,
    depth,
)
print("recovered_c_from_function =", recovered_c_from_function)





#5.相机坐标点转回世界坐标
def camera_to_world_point(point_c,R_cw,t_cw):
     """
    把相机坐标中的三维点转回世界坐标。

    参数：
        point_c: 相机坐标点，形状 (3,)
        R_cw:    世界坐标到相机坐标的旋转矩阵
        t_cw:    世界坐标到相机坐标的平移向量

    返回：
        point_w: 世界坐标点，形状 (3,)
    """
     return R_cw.T @ (point_c-t_cw)

# 使用函数恢复世界坐标
recovered_w_from_function = camera_to_world_point(
    recovered_c_from_function,
    R_cw,
    t_cw,
)
print("recovered_w_from_function =", recovered_w_from_function)






#6.相机坐标射线方向转成世界坐标射线方向
# 注意：
# - 三维点需要旋转和平移；
# - 射线方向只需要旋转，不需要平移。
# 因为方向不是位置，平移一个方向向量没有意义。
# 公式是：ray_w = R_cw.T @ ray_c
def camera_ray_to_world_ray(ray_c,R_cw):
    """
    把相机坐标中的射线方向转换到世界坐标。

    参数：
        ray_c: 相机坐标射线方向，形状 (3,)
        R_cw:  世界坐标到相机坐标的旋转矩阵

    返回：
        ray_w: 世界坐标中的射线方向，形状 (3,)
    """
    return R_cw.T @ ray_c

# 使用函数转换射线方向
ray_w_from_function = camera_ray_to_world_ray(
    ray_c_from_function,
    R_cw,
)
print("ray_w_from_function =", ray_w_from_function) 





#7.计算相机中心在世界坐标中的位置
# 相机中心在相机坐标中永远是：
# P_c = [0, 0, 0]
# 把它代入：P_c = R_cw @ P_w + t_cw
# 得到：0 = R_cw @ C_w + t_cw
# C_w = -R_cw.T @ t_cw
# 其中 C_w 就是相机中心在世界坐标中的位置。
def camera_center_in_world(R_cw,t_cw):
    """
    计算相机中心在世界坐标中的位置。

    参数：
        R_cw: 世界坐标到相机坐标的旋转矩阵
        t_cw: 世界坐标到相机坐标的平移向量

    返回：
        camera_center_w: 相机中心的世界坐标，形状 (3,)
    """
    return -R_cw.T @ t_cw

# 使用函数计算相机中心
camera_center_from_function = camera_center_in_world(R_cw, t_cw)
print("camera_center_from_function =", camera_center_from_function)




#组合函数
#1.世界点投影成像素
def project_world_point(point_w,R_cw,t_cw,K):
    """
    把世界坐标点投影到像素坐标。
    """
    point_c=world_to_camera(point_w,R_cw,t_cw)
    pixel=project_camera_to_pixel(point_c,K)
    return point_c,pixel

#用组合函数重新走一遍投影
point_c_combined,pixel_combined =project_world_point(
    point_w,
    R_cw,
    t_cw,
    K
)

print("point_c_combined =", point_c_combined)
print("pixel_combined =", pixel_combined)


#2.像素反投影成世界射线
def pixel_to_world_ray(pxiel,K,R_cw,t_cw):
    """
    把像素坐标反投影成世界坐标中的射线。

    返回：
        origin_w: 射线起点，也就是相机中心
        ray_c:    相机坐标中的射线方向
        ray_w:    世界坐标中的射线方向
    """
    #先得到相机坐标射线
    ray_c=pixel_to_camera_ray(pxiel,K)

    #射线起点是相机中心
    origin_w=camera_center_in_world(R_cw,t_cw)

    #再把方向旋转到世界坐标
    ray_w=camera_ray_to_world_ray(ray_c,R_cw)

    return origin_w,ray_c,ray_w

# 用组合函数反投影像素
origin_w_combined, ray_c_combined, ray_w_combined = pixel_to_world_ray(
    pixel_combined,
    K,
    R_cw,
    t_cw,
)
print("origin_w_combined =", origin_w_combined)
print("ray_c_combined =", ray_c_combined)
print("ray_w_combined =", ray_w_combined)



#3.像素 + 深度 → 恢复世界坐标点
def unproject_pixel_with_depth(pixel,depth,K,R_cw,t_cw):
    #像素先变成相机坐标射线
    _,ray_c,_=pixel_to_world_ray(pixel,K,R_cw,t_cw)

    #射线乘深度，得到相机坐标点
    point_c=point_on_camera_ray(ray_c,depth)

    #相机坐标点转回世界坐标
    point_w=camera_to_world_point(point_c,R_cw,t_cw)

    return point_c,point_w

# 用组合函数恢复三维点
recovered_c_final, recovered_w_final = unproject_pixel_with_depth(
    pixel_combined,
    depth,
    K,
    R_cw,
    t_cw,
)
print("recovered_c_final =", recovered_c_final)
print("recovered_w_final =", recovered_w_final)
print("original point_w =", point_w)


#重投影误差函数  检查“恢复出来的三维点重新投影后，是否还能回到原来的像素”。
def compute_reprojection_error(point_w,observed_pixel,K,R_cw,t_cw):
    #世界点重新投影成像素
    _,projected_pixel=project_world_point(
        point_w,
        R_cw,
        t_cw,
        K
    )

    #计算两个像素之间的欧氏距离
    error=np.linalg.norm(projected_pixel-observed_pixel)

    return error,projected_pixel


# 检查恢复出的世界点
error_final, projected_pixel_final = compute_reprojection_error(
    recovered_w_final,
    pixel_combined,
    K,
    R_cw,
    t_cw,
)

print("projected_pixel_final =", projected_pixel_final)
print("error_final =", error_final)