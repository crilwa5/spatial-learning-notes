from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt

#当前脚本所在目录
base_dir=Path(__file__).parent

#视频抽帧后的图片序列
sequence_dir=base_dir/"sequence"




#读取图片（避免中文路径问题）
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
         new_width = int(width * scale)
         new_height = int(height * scale)
         return cv2.resize(
            image,
            (new_width, new_height),
            interpolation=cv2.INTER_AREA,
        )
     
     return image








#ORB 特征匹配  返回特征点和可靠匹配
def match_orb_features(gray1,gray2):
    """
    对两张灰度图提取 ORB 特征，并做 Lowe ratio test 筛选匹配。
    """
    orb=cv2.ORB_create(nfeatures=1000)

    keypoints1,descriptors1=orb.detectAndCompute(gray1,None)
    keypoints2,descriptors2=orb.detectAndCompute(gray2,None)

    if descriptors1 is None or descriptors2 is None:
        raise RuntimeError("没有检测到足够的ORB描述子")

    bf=cv2.BFMatcher(cv2.NORM_HAMMING)
    raw_matches=bf.knnMatch(descriptors1,descriptors2,k=2)

    good_matches=[]

    for pair in raw_matches:
        if len(pair)<2:
            continue

        best_match,second_match=pair

        #最佳匹配明显优于第二匹配时，才保留
        if best_match.distance<0.75*second_match.distance:
            good_matches.append(best_match)

    return keypoints1,keypoints2,good_matches





#核心函数 estimate_relative_pose  
#读两张图 → ORB 匹配 → RANSAC 筛内点 → 估计基础矩阵 F → 估计本质矩阵 E → 恢复 R、t 
def estimate_relative_pose(image1_path,image2_path):
    """
    对两张图片估计相对相机运动。

    返回：
        R: 第一帧相机坐标 -> 第二帧相机坐标的旋转
        t: 对应的平移方向，长度归一化
        stats: 匹配和内点统计
    """
    # 1）读图 + 缩放 + 转灰度
    img1=read_image_unicode(image1_path)
    img2=read_image_unicode(image2_path)

    img1=resize_keep_ratio(img1)
    img2=resize_keep_ratio(img2)

    gray1=cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    gray2=cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)

    # 2）ORB 匹配
    keypoints1,keypoints2,good_matches=match_orb_features(
        gray1,
        gray2
    )

    if len(good_matches)<8:
        raise RuntimeError(
            f"匹配点太少，无法估计位姿:{len(good_matches)}"
        )

    # 3）提取匹配点坐标
    points1=np.float32(
        [keypoints1[m.queryIdx].pt for m in good_matches]
    )
    points2=np.float32(
        [keypoints2[m.trainIdx].pt for m in good_matches]
    )

    # 4）RANSAC 计算基础矩阵 F
    #基础矩阵用于筛选对极几何一致的内点
    F,inlier_mask=cv2.findFundamentalMat(
        points1,
        points2,
        cv2.FM_RANSAC,
        1.0,
        0.999,
        2000
    )

    if F is None or inlier_mask is None:
        raise RuntimeError("基础矩阵估计失败")

    inlier_mask=inlier_mask.ravel().astype(bool)

    points1_inliers=points1[inlier_mask]
    points2_inliers=points2[inlier_mask]

    # 5）构造近似内参
    #近似内参：真实项目使用相机标定结果
    height,width=gray1.shape
    focal=1.0*max(width,height)

    K_approx=np.array(
        [
            [focal,0.0,width/2.0],
            [0.0,focal,height/2.0],
            [0.0,0.0,1.0]
        ],
        dtype=float
    )

    # 6）RANSAC 计算本质矩阵 E
    E,essential_mask=cv2.findEssentialMat(
        points1_inliers,
        points2_inliers,
        K_approx,
        method=cv2.RANSAC,
        prob=0.999,
        threshold=1.0
    )

    if E is None:
        raise RuntimeError("本质矩阵估计失败")

    # 7）从本质矩阵恢复相对旋转和平移 R，t
    num_pose_inliers,R,t,pose_mask=cv2.recoverPose(
        E,
        points1_inliers,
        points2_inliers,
        K_approx
    )

    # 8）统计信息
    stats={
        "good_matches":len(good_matches),
        "ransac_inliers":int(inlier_mask.sum()),
        "pose_inliers":int(num_pose_inliers),
        "K":K_approx,
        "E":E,
        "F":F,
        # 保存 RANSAC 筛选后的二维匹配点
        "points1_inliers": points1_inliers,  #第一张图中的二维点
        "points2_inliers": points2_inliers,  #第二张图中的二维点
    }

    return R,t,stats


#main()
# 从一堆图片里每隔 8 帧取一张 → 两两计算相机怎么移动 → 把这些小运动累积起来 → 得到一条完整的相机轨迹

# 核心思路
# 假设你有三帧：0.jpg、8.jpg、16.jpg。
# 第 1 步：估计相对运动
# 用 0.jpg 和 8.jpg 估计：相机从 0 到 8 怎么动的。
# 用 8.jpg 和 16.jpg 估计：相机从 8 到 16 怎么动的。

# 第 2 步：累积位姿
# 相机 0 放在世界原点。
# 相机 8 的位置 = 相机 0 的位置 + 从 0 到 8 的运动。
# 相机 16 的位置 = 相机 8 的位置 + 从 8 到 16 的运动。

# 第 3 步：画轨迹
# 把所有相机中心连起来，就是相机走过的路径。
def main():
    # #先用序列中的前两帧测试
    # image1_path=sequence_dir/"0.jpg"
    # image2_path=sequence_dir/"8.jpg"

    # R,t,stats=estimate_relative_pose(
    #     image1_path,
    #     image2_path
    # )

    # print("image1:", image1_path)
    # print("image2:", image2_path)
    # print("good matches =", stats["good_matches"])
    # print("ransac inliers =", stats["ransac_inliers"])
    # print("pose inliers =", stats["pose_inliers"])

    # print("R =")
    # print(R)

    # print("t =")
    # print(t)

    # C2_in_1=-R.T @ t

    # print("第二台相机在第一台相机坐标中的位置 C2_in_1 =")
    # print(C2_in_1)

    # 相邻两帧：recoverPose 得到 R、t
    # 把关系反过来：T_curr_prev = inverse(T_prev_curr)
    # 再把多个相对位姿累积：T_wc = T_wc @ T_curr_prev
    # 其中 T_wc 可以理解为：相机坐标 → 世界坐标
    # 它的平移部分就是相机中心在世界中的位置



    # 第一部分：找到所有图片  把 sequence 文件夹里面所有 .jpg 图片找出来，并按照数字排序
    frame_paths=sorted(
        sequence_dir.glob("*.jpg"),#找出所有 .jpg 文件
        #path.stem 是文件名去掉后缀的部分。比如 0.jpg 的 stem 是 "0"。int("0") 就是数字 0。
        key=lambda path: int(path.stem)  #按照图片文件名里的数字排序
    )





    #第二部分：每隔 8 帧取一张    0,8,16,24...
    frame_gap=8
    selected_frames=frame_paths[::frame_gap]  #从头开始，每隔 8 个元素取一个

    #把选中的图片打印出来
    print("选择的帧:")
    for path in selected_frames:
        print(" ",path.name)





    #第三部分：让第一台相机站在原点 
    # 规定第一张图片对应的相机就是世界坐标系的原点 第0帧相机位置 = (0, 0, 0)
    #T_wc表示相机坐标 -> 世界坐标
    T_wc=np.eye(4)  #4x4 单位矩阵






    #第四部分：记录第一台相机的位置
    trajectory=[T_wc[:3,3].copy()]
    # T_wc[:3, 3]就是：取 4×4 矩阵的第 4 列前三个数字。
    # 也就是：[ x ][ y ][ z ]这三个数字代表：相机中心的位置。
    # 一开始：T_wc =
    # [1 0 0 0]
    # [0 1 0 0]
    # [0 0 1 0]
    # [0 0 0 1]
    # 所以：T_wc[:3,3]得到：[0, 0, 0]因此：trajectory = [[0,0,0]]
    # 意思：先把第一台相机的位置记下来。






    #把相邻的图片一对一对拿出来处理  0-8 8-16... selectedframes里面是0,8,16...
    for index in range(len(selected_frames)-1):
        #取出当前两张图片
        image1_path=selected_frames[index]
        image2_path=selected_frames[index+1]

        print()
        print("处理:",image1_path.name,"->",image2_path.name)

        try:
            #调用前面的 estimate_relative_pose()
            # R代表：旋转,也就是：相机方向变了多少。
            # 而：t代表：平移方向,也就是：相机往哪里移动
            
            R,t,stats=estimate_relative_pose(
                image1_path,
                image2_path
            )
        except Exception as error:
            print("跳过这一对，原因:",error)
            continue

        #stats 是干什么的？
        # print("good matches =", stats["good_matches"])
        # print("ransac inliers =", stats["ransac_inliers"])
        # print("pose inliers =", stats["pose_inliers"])
        # 就是在看：“这一次计算到底靠不靠谱？”
        # 例如：good matches = 150
        # ransac inliers = 100
        # pose inliers = 85
        # 说明：匹配了很多点↓其中很多符合几何关系↓最后很多点支持这个相机运动估计
        print("good matches =",stats["good_matches"])
        print("ransac inliers =",stats["ransac_inliers"])
        print("pose inliers =",stats["pose_inliers"])

        #内点太少，认为这一对不可靠  如果这两张图匹配得太差，我不相信这一次计算，直接跳过
        if stats["pose_inliers"]<30:
            print("pose inliers 太少，跳过这一对")
            continue




        #把刚才的 R 和 t 装进一个 4×4 矩阵  位姿/坐标变换矩阵
        #recoverPose 得到的是：上一帧相机坐标->当前帧相机坐标
        # T_prev_curr 描述“上一帧和当前帧之间关系”的东西
        T_prev_curr=np.eye(4)
        T_prev_curr[:3,:3]=R
        T_prev_curr[:3,3]=t.ravel()

        #为了累积到世界轨迹，取它的逆：当前帧相机坐标->上一帧相机坐标

        # recoverPose() 给你的关系是：上一帧 → 当前帧
        # 也就是：prev → curr
        # 但是你现在要把相机位置放到：世界坐标系
        # 所以你代码选择把它反过来：curr → prev
        # 于是：inverse()就是：把这个坐标变换方向反过来

        # 用箭头理解 inv() 最简单
        # 原来：A → B
        # 求逆之后：B → A
        # 所以：T_curr_prev = inv(T_prev_curr)
        # 可以先理解成：原来：上一帧 → 当前帧
        # 取逆：当前帧 → 上一帧
        T_curr_prev=np.linalg.inv(T_prev_curr)






        #累积位姿  累积轨迹
        # T_wc 不断更新：第一次：T_wc = 第一次运动
        # 第二次：T_wc = 第一次运动 × 第二次运动
        # 第三次：T_wc = 前面的结果 × 第三次运动
        T_wc=T_wc @ T_curr_prev  #把这一次的新运动接到之前已经走过的路线后面


        #相机中心就是T_wc的平移部分  得到当前相机位置
        camera_center_w=T_wc[:3,3].copy()  # T_wc[:3,3] 取x，y，z  

        #把这一次相机的位置加入轨迹列表
        trajectory.append(camera_center_w)

        print("当前相机中心:",np.round(camera_center_w,4)) #把当前相机中心的坐标，四舍五入保留 4 位小数后打印出来

    #变成 NumPy 数组
    trajectory=np.asarray(trajectory)

    print()
    print("轨迹 shape =",trajectory.shape)
    print("轨迹:")
    print(np.round(trajectory,4))






    #把这条 4 个点的相机轨迹画出来
    # 创建结果目录
    result_dir = base_dir / "results"
    result_dir.mkdir(exist_ok=True)

    # 创建三维图
    fig = plt.figure(figsize=(8, 6))  #宽，高
    ax = fig.add_subplot(111, projection="3d")  #三维坐标系


    # plot()主要是：画线。
    # scatter()主要是：画一个点。

    # 画相机轨迹
    ax.plot(
        trajectory[:, 0],  #所有相机位置的 X  所有第一列
        trajectory[:, 1],  #所有相机位置的 y  所有第二列
        trajectory[:, 2],  #所有相机位置的 z  所有第三列
        marker="o",   #每一个相机位置画一个圆点
        linewidth=2,   #轨迹线画粗一点
        label="VO trajectory",  #线的名字
    )

    # 标出起点和终点
    #把第一个相机位置单独画出来 起点
    ax.scatter(
        trajectory[0, 0],
        trajectory[0, 1],
        trajectory[0, 2],
        color="green",
        s=80,
        label="start",
    )

    #标记终点
    ax.scatter(
        trajectory[-1, 0],
        trajectory[-1, 1],
        trajectory[-1, 2],
        color="red",
        s=80,
        label="end",
    )

    # 坐标轴名称
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    ax.set_title("Simplified VO Camera Trajectory")

    # 让三个轴尽量显示得均匀  让 X、Y、Z 三个方向的显示比例尽量合理
    ax.set_box_aspect(
        [
            np.ptp(trajectory[:, 0]) + 1e-6,
            np.ptp(trajectory[:, 1]) + 1e-6,
            np.ptp(trajectory[:, 2]) + 1e-6,
        ]
    )

    ax.legend()    #显示图例

    # 保存图片
    plot_path = result_dir / "vo_trajectory.png"
    plt.savefig(plot_path, dpi=200, bbox_inches="tight")

    print("轨迹图已保存:", plot_path)

    # 打开图形窗口
    # plt.show()
    plt.close(fig)








    # 理解单目 VO 的尺度不确定性 
    # 轨迹不能直接说是“多少米”？recoverPose 返回的 t 只有方向，长度被归一化了
    # 什么叫“归一化”？
    # 你看到：t 的长度被归一化其实就是把：t强行缩放成：长度 = 1
    # 例如原来：t = [3, 0, 0]长度是：3
    # 归一化之后：t = [1, 0, 0]长度变成：1
    # 但是方向没变。所以：不是说真实世界里相机真的只走了 1 米。
    # 而是：程序暂时只保留“方向”，把真实距离缩掉了。
    #把归一化轨迹整体放大2倍
    #注意：这不是恢复出真实尺度，只是展示“尺度可以任意变化”
    scale_factor=2.0
    trajectory_scaled=trajectory*scale_factor

    print("原始轨迹:")
    print(np.round(trajectory,4))

    print("放大2倍后的轨迹:")
    print(np.round(trajectory_scaled,4))

    # 保存两份轨迹，观察它们只是尺度不同
    normalized_path = result_dir / "trajectory_normalized.txt"
    scaled_path = result_dir / "trajectory_scaled.txt"
    np.savetxt(
        normalized_path,
        trajectory,
        fmt="%.6f",
    )
    np.savetxt(
        scaled_path,
        trajectory_scaled,
        fmt="%.6f",
    )
    
    print("归一化轨迹已保存:", normalized_path)
    print("放大后的轨迹已保存:", scaled_path)








    #把匹配点三角化，恢复出三维点，得到最基础的三维地图
    # 相机 1 像素 + 相机 2 像素 + 两个相机位姿
    # → 两条射线
    # → 在三维空间中求交点
    # → 一个三维点
    # 多组匹配点一起三角化，就得到点云：points3d.shape = (M, 3)
    #使用0和8进行三角化
    #第 1 步：选择两张图片
    tri_image1=selected_frames[0]
    tri_image2=selected_frames[1]

    #第 2 步：重新估计两个相机之间的运动
    R_tri,t_tri,stats_tri=estimate_relative_pose(
        tri_image1,
        tri_image2
    )

    #第 3 步：拿到相机内参 K  主要描述：相机镜头是怎么看三维世界投影成二维图片的
    K_tri=stats_tri["K"]

    #第 4 步：拿到两张图片中的匹配点
    points1_tri=stats_tri["points1_inliers"]
    points2_tri=stats_tri["points2_inliers"]

    #第一台相机的投影矩阵
    #世界坐标就是第一台相机的坐标
    #第 5 步：创建第一个相机的投影矩阵
    #np.hstack()意思：横着拼起来。
    # 变成： [1 0 0 | 0]
    #       [0 1 0 | 0]
    #       [0 0 1 | 0]
    # 也就是：[ I | 0 ]
    P1=K_tri @ np.hstack(
        [
            np.eye(3),  #3×3 单位矩阵
            np.zeros((3,1))
        ]
    )
    # 再乘 K  所以：P1 = K_tri @ [I | 0]
    # 得到：P1 = K[I|0]这个 P1 就是：第一个相机的投影矩阵
    # 为什么第一个相机是 [I | 0]？因为我们人为规定：第一个相机就是世界坐标系的原点。
    # 也就是：世界坐标系 ↓第一个相机
    # 我们直接说：相机1的位置 = 世界原点

    #第 6 步：创建第二个相机的投影矩阵
    # P2=[R,t]
    # [R | t]表示：第二个相机相对于第一个相机的旋转和平移关系
    P2=K_tri @ np.hstack(
        [
            R_tri,
            t_tri
        ]
    )

    #为什么需要 P1 和 P2？
    # 因为三角化函数需要知道：这两个相机分别站在哪里、朝哪个方向看


    #三角化
    #cv2.triangulatepoints需要(2,N)的二维点
    points3d_h=cv2.triangulatePoints(
        P1,
        P2,
        points1_tri.T,
        points2_tri.T
    )
    # 这里的 _h：h = homogeneous，齐次坐标。
    # 它不是普通的：x，y，z而是：x，y，z，w
    # 所以它的 shape 通常是：(4, M)


    #从齐次坐标转成普通三维坐标
    points3d=(
        points3d_h[:3]/points3d_h[3]  #前三行除以第四行得到前三维的，然后转置回来
    ).T  #(M, 3)

    #检查点是否在两个相机正前方
    #因为我们把第一个相机作为世界坐标系，所以这些三维点现在可以直接看成第一个相机坐标系里的点
    points_in_camera1=points3d

    #把这些三维点从第一个相机坐标系转换到第二个相机坐标系
    points_in_camera2=(
        R_tri @ points3d.T + t_tri
    ).T

    #只留下同时在两个相机前面的三维点
    valid_mask=(
        (points_in_camera1[:,2]>0) #z都大于0
        &(points_in_camera2[:,2]>0)
    )

    #最后真正留下三维点
    points3d_valid=points3d[valid_mask]

    print()
    print("三角化前的匹配点数 =",len(points1_tri))
    print("三角化后有效三维点数 =",len(points3d_valid))
    print("points3d_valid shape =",points3d_valid.shape)

    #保存三维点
    points3d_path=result_dir/"triangulated_points.xyz"
    np.savetxt(
        points3d_path,
        points3d_valid,
        fmt="%.6f"
    )

    print("三角化点云已保存:",points3d_path)







    #把三角化得到的三维点画出来
    fig2=plt.figure(figsize=(8,6))
    ax2=fig2.add_subplot(111,projection="3d")

    #画三维点
    ax2.scatter(
        points3d_valid[:,0],
        points3d_valid[:,1],
        points3d_valid[:,2],
        s=8,
        c=points3d_valid[:,2],
        cmap="viridis",
        label="triangulated points"
    )

    #相机轨迹也画上
    ax2.plot(
        trajectory[:,0],
        trajectory[:,1],
        trajectory[:,2],
        color="red",
        marker="o",
        linewidth=2,
        label="camera trajectory"
    )

    #标出第一台相机的位置
    ax2.scatter(
        0.0,
        0.0,
        0.0,
        color="green",
        s=100,
        label="camera 1"
    )

    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")
    ax2.set_zlabel("Z")
    ax2.set_title("Triangulated 3D Points")
    
    # 让三个坐标轴比例尽量接近真实比例
    mins=points3d_valid.min(axis=0)
    maxs=points3d_valid.max(axis=0)
    ranges=np.maximum(maxs-mins,1e-6)

    ax2.set_box_aspect(ranges)
    ax2.legend()

    #保存图片
    points_plot_path=result_dir/"triangulated_points.png"
    plt.savefig(
        points_plot_path,
        dpi=200,
        bbox_inches="tight"
    )

    print("三维点可视化已保存:",points_plot_path)

    #不阻塞窗口
    plt.close(fig2)









    #计算重投影误差
    # 三角化得到的三维点
    # → 用第二个相机位姿重新投影回图片
    # → 和第二个相机里真实观测到的像素比较
    # → 计算误差
    # 误差越小，说明相机位姿和三维点越一致

    #只保留和points3d_valid 对应的第二帧观测点
    points2_valid=points2_tri[valid_mask] #N,2

    #把三维点变成齐次坐标，形状从(N,3)->(N,4)
    #why?因为投影矩阵是3x4的，到时候要进行@，所以要扩到N,4
    points3d_h=np.hstack(
        [
            points3d_valid,
            np.ones((len(points3d_valid),1))  #用1来补最后一列
        ]
    )

    #第二台相机的投影矩阵 K[R|t]  3x3 @ 3x4 -> 3x4
    P2_reproject=K_tri @ np.hstack(
        [
            R_tri,
            t_tri
        ]
    )

    #把三维点投影到第二张图片  K[R|t]得到的是3x4，3x4 @ 4xN ->所以再转置一下为N,3
    projected_h=(P2_reproject @ points3d_h.T).T

    #齐次坐标->普通像素坐标   像素坐标是u，v，所以还要除以第三列
    projected_pixels=( #N,2
        projected_h[:,:2]/projected_h[:,2:3]
    )

    #计算每个点的重投影误差
    reprojection_errors=np.linalg.norm( #N,
        projected_pixels-points2_valid,
        axis=1  #按每行来计算误差
    )

    print("重投影误差平均值 =",reprojection_errors.mean())
    print("重投影误差中位数 =",np.median(reprojection_errors))
    print("重投影误差最大 =",reprojection_errors.max())

    # 保存误差
    error_path = result_dir / "reprojection_errors.txt"
    np.savetxt(
        error_path,
        reprojection_errors,
        fmt="%.6f",
    )
    
    print("重投影误差已保存:", error_path)

    # 绘制重投影误差直方图
    fig3 = plt.figure(figsize=(8, 5))
    ax3 = fig3.add_subplot(111)

    #画直方图
    ax3.hist(
        reprojection_errors,
        bins=30,  #把误差范围分成 30 个小区间
        color="steelblue",
        edgecolor="black",
    )
    
    ax3.set_xlabel("Reprojection error (pixels)")
    ax3.set_ylabel("Number of points")
    ax3.set_title("Reprojection Error Distribution")
    
    hist_path = result_dir / "reprojection_error_hist.png"
    plt.savefig(hist_path, dpi=200, bbox_inches="tight")
    
    print("重投影误差图已保存:", hist_path)
    plt.close(fig3)














    #只优化第二个相机位姿，看误差能不能进一步降低
    # 先有一个初始位姿 → 根据 3D 点和真实 2D 点重新调整位姿 → 看误差有没有下降。”
    # 你可以把它理解成：前面是“猜一个相机位置”，这里是“拿答案反过来微调这个相机位置”
    # 这一步用的是 solvePnP：
    # 已知三维点
    # 已知它们在第二张图中的像素位置
    # 已知相机内参
    # → 调整第二台相机的 R、t
    # → 让重投影误差变小

    # solvePnP 到底是什么？
    # 只需要记：
    # 已知：3D点↓[X,Y,Z]
    # 已知：图片中的位置↓[u,v]
    # 已知：相机内参 K
    # 求：相机在哪里？↓R + t
    # 所以：solvePnP 是一个“根据 3D 点和它们在图片中的位置，反过来求相机位姿”的方法。





    # 1）把初始 R、t 转成 PnP 需要的形式
    # solvePnP 使用旋转向量rvec和平移向量tvec
    # solvePnP想要什么？它想要：rvec旋转向量。形状通常是：(3,1)
    # 可以简单理解成：旋转轴 + 旋转角度
    # 现在你只需要理解：它是 R 的另一种表示方式。 Rodrigues实现

    # 先把原来的R转成rvec  3x3->3x1
    rvec_init,_=cv2.Rodrigues(R_tri)

    #平移向量保持为(3,1)
    tvec_init=t_tri.reshape(3,1)

    print("rvec_init shape =",rvec_init.shape) #3x1
    print("tvec_init shape =",tvec_init.shape) #3x1

    #第二台相机的初始位姿
    # 旋转：rvec_init
    # 平移：tvec_init


    # 2)使用 solvePnP 优化第二台相机位姿
    # 使用已知三维点和第二帧二维观测，优化第二台相机
    success,rvec_refined,t_refined=cv2.solvePnP(
        #3D点  N,3->N,1,3
        points3d_valid.reshape(-1,1,3), 

        #2D像素 N,2->N,1,2
        points2_valid.reshape(-1,1,2),

        #相机内参 3x3
        K_tri,

        #畸变系数
        None,

        #初始旋转
        rvec_init.copy(),

        #初始平移
        tvec_init.copy(),

        # 告诉 solvePnP：我已经给你一个初始的 R 和 t，不要从完全随机/默认的地方开始，
        # 直接从我这个初始位姿附近优化。
        useExtrinsicGuess=True,

        #使用迭代优化方法
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not success:
        raise RuntimeError("solvePnP 位姿优化失败")

    #把优化后的旋转向量转回旋转矩阵 3x1->3x3
    R_refined,_=cv2.Rodrigues(rvec_refined)

    print("优化后的 R shape =",R_refined.shape) #3x3
    print("优化后的 t shape =",t_refined.shape) #3x1
    print("优化后的 t =")
    print(t_refined)


    # 3)使用优化后的相机位姿重新投影
    #投影矩阵
    P2_refined = K_tri @ np.hstack(
        [
            R_refined,
            t_refined,
        ]
    )

    #重新投影
    projected_refined_h = (
        P2_refined @ points3d_h.T
    ).T

    #变成普通像素
    projected_refined_pixels = (
        projected_refined_h[:, :2]
        / projected_refined_h[:, 2:3]
    )
    
    optimized_errors = np.linalg.norm(
        projected_refined_pixels - points2_valid,
        axis=1,
    )
    
    print()
    print("优化前平均重投影误差 =", reprojection_errors.mean())
    print("优化后平均重投影误差 =", optimized_errors.mean())
    
    print("优化前中位误差 =", np.median(reprojection_errors))
    print("优化后中位误差 =", np.median(optimized_errors))
    
    print("优化前最大误差 =", reprojection_errors.max())
    print("优化后最大误差 =", optimized_errors.max())


    # 4)保存优化后的误差
    optimized_error_path = result_dir / "reprojection_errors_refined.txt"
    np.savetxt(
        optimized_error_path,
        optimized_errors,
        fmt="%.6f",
    )
    
    print("优化后误差已保存:", optimized_error_path)
    
    # 保存优化后的相对位姿
    pose_path = result_dir / "refined_relative_pose.npz"
    np.savez(
        pose_path,
        R=R_refined,
        t=t_refined,
    )
    
    print("优化后的相对位姿已保存:", pose_path)








 
    #简化版 BA  真正的 BA 是把所有相机位姿和所有 3D 点一起放进一个整体优化问题
    # 交替做：固定三维点 → 优化相机位姿
    # 固定相机位姿 → 重新三角化三维点
    # 重复几轮

    # 保存与当前三维点对应的第一帧匹配点
    points1_valid = points1_tri[valid_mask]  #N,2

    # 1）初始化交替优化变量
    R_current = R_refined.copy()  #3x3
    t_current = t_refined.copy()  #3x1
    
    points3d_current = points3d_valid.copy()  # N,3
    points1_current = points1_valid.copy()  #第一帧二维点 N,2
    points2_current = points2_valid.copy()  #第二帧二维点 N,2
    
    # 记录每一轮的重投影误差
    error_history = []

    # 2）交替优化循环
    #重复若干轮
    for iteration in range(5):
        #当前位姿下的重投影
        # 第一部分：先计算当前误差
        points3d_h_current=np.hstack( #N,4  扩成四维来计算
            [
                points3d_current,  #N,3
                np.ones((len(points3d_current),1)) #加一个1
            ]
        )

        #构造当前投影矩阵
        P2_current=K_tri @ np.hstack( #3x3 @ 3x4 -> 3x4
            [
                R_current, #3x3
                t_current #3x1
            ]
        )

        #当前三维点重新投影
        projected_current_h=( # 3x4 @ 4xN -> 3xN -> N,3
            P2_current @ points3d_h_current.T
        ).T

        #转成普通像素
        projected_current_pixels=(#取前两列除以第三列 ->N,2
            projected_current_h[:,:2]
            /projected_current_h[:,2:3]
        )

        #计算当前误差
        current_errors=np.linalg.norm(
            projected_current_pixels-points2_current,
            axis=1
        )

        #计算这一轮平均误差
        mean_error=float(current_errors.mean())
        error_history.append(mean_error)

        #打印这一轮
        print(
            f"迭代 {iteration}: "
            f"点数={len(points3d_current)}, "
            f"平均误差={mean_error:.6f}"
        )



        #第一步：固定 3D 点，优化相机  
        #先把 R 转成旋转向量 3x3->3x1
        rvec_current,_=cv2.Rodrigues(R_current) 

        #solvePnP
        success,rvec_new,tvec_new=cv2.solvePnP(
            points3d_current.reshape(-1,1,3), #N,1,3
            points2_current.reshape(-1,1,2),  #N,1,2
            K_tri,
            None,
            rvec_current.copy(),
            t_current.reshape(3,1).copy(),
            useExtrinsicGuess=True,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        if not success:
            print("solvePnP 优化失败，停止迭代")
            break

        #旋转向量变回旋转矩阵 3,1->3x3
        R_current,_=cv2.Rodrigues(rvec_new)
        #更新平移矩阵
        t_current=tvec_new

        #这一步：相机位姿被更新了


        # 第二步：固定相机，重新三角化
        # 第一台相机就是世界坐标原点 0,0,0
        P1_current=K_tri @ np.hstack(
            [
                np.eye(3),
                np.zeros((3,1))
            ]
        )

        #第二台相机用刚才优化后的位姿
        P2_new=K_tri @ np.hstack(
            [
                R_current,
                t_current
            ]
        )

        #重新三角化
        triangulated_h=cv2.triangulatePoints( #返回(4,N) 四维齐次坐标 [x][y][z][w]
            P1_current,
            P2_new,
            points1_current.T,
            points2_current.T
        )

        #转换成普通 3D 坐标
        triangulated_points=( #4,N->N,3
            #取前三行除以第四行
            triangulated_h[:3]/triangulated_h[3]
        ).T

        #检查点是否仍在两个相机正前方
        points_in_camera1=triangulated_points
        points_in_camera2=( #把这些重新三角化的世界坐标点转换到第二台相机坐标系
            R_current @ triangulated_points.T
            + t_current
        ).T

        valid_current=(
            (points_in_camera1[:,2]>0)
            &(points_in_camera2[:,2]>0)
        )

        #更新当前三维点，并同步筛选三维匹配点
        points3d_current=triangulated_points[valid_current]
        points1_current=points1_current[valid_current]
        points2_current=points2_current[valid_current]


    print("交替优化结束")
    print("误差变化:")
    print([round(value,6) for value in error_history])


    

    # 3)保存优化后的三维点
    ba_points_path=result_dir/"ba_refined_points.xyz"
    np.savetxt(
        ba_points_path,
        points3d_current,
        fmt="%.6f"
    )

    # 4)保存优化后的相机位姿
    ba_pose_path=result_dir/"ba_refined_pose.npz"
    np.savez(
        ba_pose_path,
        R=R_current,
        t=t_current
    )

    print("BA 优化点云已保存:",ba_points_path)
    print("BA 优化位姿已保存:",ba_pose_path)



    # 5)画误差下降曲线
    fig4 = plt.figure(figsize=(7, 5))
    ax4 = fig4.add_subplot(111)
    ax4.plot(
        range(len(error_history)),
        error_history,
        marker="o",
        color="purple",
    )
    
    ax4.set_xlabel("Iteration")
    ax4.set_ylabel("Mean reprojection error (pixels)")
    ax4.set_title("Simplified BA Error Curve")
    
    error_curve_path = result_dir / "ba_error_curve.png"
    plt.savefig(error_curve_path, dpi=200, bbox_inches="tight")
    
    print("BA 误差曲线已保存:", error_curve_path)
    plt.close(fig4)












if __name__ == "__main__":
    main()


# 单目相机通常无法仅靠两张普通图片确定真实尺度
# 那什么时候才能知道“真实距离”？需要额外的信息。
# 方法 1：双目相机
# 两只眼睛：
# 👁️       👁️
#  \       /
#   \     /
#    🚗
# 两个相机之间有一个已知距离。这个距离相当于尺子。
# 于是可以恢复真实尺度。

# 方法 2：RGB-D 相机
# 比如：RGB相机 + 深度相机
# 直接获得：这个点距离相机 2.3米
# 那么尺度就有了。

# 方法 3：IMU
# 现在正在学习的 VIO：
# 相机 👀
#    +
# IMU 📱
# IMU 提供额外的运动信息。
# 在合理的初始化和模型条件下，可以帮助恢复尺度。

# 方法 4：GPS / 已知尺寸
# 比如知道：两个路标之间 = 10米
# 那么可以拿它当尺子。





# ① 当前3D点
#       ↓
# ② 固定3D点
#    solvePnP
#       ↓
#    更新相机 R,t
#       ↓

# ③ 固定相机
#    triangulatePoints
#       ↓
#    更新3D点
#       ↓

# ④ 重复

# ① 三角化
# 2D + 2D
#  ↓
# 3D
# “根据两个相机看到的位置，猜 3D 点。”

# ② solvePnP
# 3D + 2D
#  ↓
# R + t
# “3D 点已经有了，根据图片中的位置调整相机。”

# ③ 简化 BA
# 3D + 2D
#  ↓
# 调相机
#  ↓
# 重新算3D
#  ↓
# 再调相机
#  ↓
# 再算3D
# ...
# “相机和 3D 点轮流调整，让它们彼此更一致。”
     