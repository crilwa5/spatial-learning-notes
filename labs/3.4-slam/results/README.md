# 3.4 SLAM 实验结果记录

本文件记录本次最小实验的真实运行结果和结果文件含义。

---

## 1. 图片读取结果

原始图片：

```text
0001.JPG: (5712, 4284, 3)
0002.JPG: (5712, 4284, 3)
```

缩放后：

```text
img1_small: (1280, 960, 3)
img2_small: (1280, 960, 3)
gray1: (1280, 960)
gray2: (1280, 960)
```

---

## 2. ORB 特征

```text
keypoints1 = 1000
keypoints2 = 1000
descriptors1 = (1000, 32)
descriptors2 = (1000, 32)
```

解释：

- 每张图最多保留 1000 个 ORB 特征；
- 每个描述子是 32 字节；
- 描述子用于判断不同图像中的特征是否相似。

---

## 3. 特征匹配和 RANSAC

初始匹配：

```text
raw matches = 1000
good matches = 186
```

使用基础矩阵 F 做 RANSAC 后：

```text
RANSAC 前匹配数 = 186
RANSAC 内点数 = 121
RANSAC 外点数 = 65
F shape = (3, 3)
```

结果说明：

- `good matches` 中仍混有错误匹配；
- RANSAC 利用几何一致性去掉了 65 个外点；
- 剩余 121 个点更可靠。

结果文件：

```text
results/feature_matches.jpg
results/feature_matches_ransac.jpg
```

---

## 4. 相邻两帧为什么 pose_inliers = 0？

使用 `0.jpg -> 1.jpg` 时：

```text
pose_inliers = 0
```

原因：

- 两帧间隔太小；
- 相机位移和视差不足；
- 本质矩阵几何条件不稳定；
- 不是 `recoverPose` 代码本身报错，而是输入基线不合适。

改用：

```text
0.jpg -> 8.jpg
```

后结果正常：

```text
good matches = 475
RANSAC 内点 = 331
pose_inliers = 331
```

这是一个重要的实际经验：

> VO 不是相邻帧越近越好，两帧之间需要足够的相机位移和视差。

---

## 5. 相对相机运动

`0.jpg -> 8.jpg` 的结果：

```text
t ≈ [0.9944, 0.0779, 0.0715]
```

第二台相机在第一台相机坐标中的位置：

```text
C2_in_1 ≈ [-0.9946, -0.0868, 0.0569]
```

注意：

- `t` 只有方向，没有真实米数；
- `C2_in_1` 也处于归一化尺度；
- 单目 VO 的尺度无法只靠两帧图像恢复。

---

## 6. 多帧相机轨迹

选取帧：

```text
0.jpg
8.jpg
16.jpg
24.jpg
```

每一对的 pose inliers：

```text
0 -> 8: 331
8 -> 16: 112
16 -> 24: 89
```

最终轨迹：

```text
trajectory.shape = (4, 3)

[ 0.0000,  0.0000, 0.0000]
[-0.9946, -0.0868, 0.0569]
[-1.9188, -0.1962, 0.4228]
[-2.6457, -0.1841, 1.1095]
```

结果文件：

```text
results/vo_trajectory.png
results/trajectory_normalized.txt
results/trajectory_scaled.txt
```

---

## 7. 尺度不确定性演示

原轨迹乘以 2 后：

```text
[-1.9892, -0.1736, 0.1139]
[-3.8376, -0.3924, 0.8456]
[-5.2914, -0.3682, 2.2189]
```

这证明：

```text
同一组图像可以对应多条仅尺度不同的轨迹
```

单目视觉不能仅靠图像恢复绝对尺度。

---

## 8. 三角化三维点

使用 `0.jpg` 和 `8.jpg`：

```text
三角化前匹配点数 = 331
三角化后有效三维点数 = 331
points3d_valid.shape = (331, 3)
```

结果文件：

```text
results/triangulated_points.xyz
results/triangulated_points.png
```

这些三维点位于：

```text
第一台相机坐标系
归一化尺度
```

---

## 9. 重投影误差

初始三角化和相机位姿：

```text
平均误差 = 0.322116
中位误差 = 0.274452
最大误差 = 2.107556
```

结果文件：

```text
results/reprojection_errors.txt
results/reprojection_error_hist.png
```

解释：

- 大部分误差很小，说明匹配、位姿和三角化总体一致；
- 少数点误差较大，可能来自误匹配、小视差或近似内参。

---

## 10. solvePnP 位姿优化

固定三维点，优化第二台相机位姿后：

```text
平均误差 = 0.301731
中位误差 = 0.257380
最大误差 = 1.693901
```

结果文件：

```text
results/reprojection_errors_refined.txt
results/refined_relative_pose.npz
```

结论：

> 调整相机位姿可以降低重投影误差。

---

## 11. 简化交替 BA

交替执行：

```text
固定三维点，优化相机位姿
固定相机位姿，重新三角化三维点
```

平均误差变化：

```text
迭代 0: 0.301731
迭代 1: 0.286239
迭代 2: 0.266527
迭代 3: 0.256684
迭代 4: 0.252148
```

结果文件：

```text
results/ba_refined_points.xyz
results/ba_refined_pose.npz
results/ba_error_curve.png
```

结论：

> 相机位姿和三维点互相影响，联合检查并反复优化可以让重投影误差下降。

注意：

> 这是简化的交替优化，不是完整工业级 BA。

---

## 12. 回环修正演示

模拟闭合轨迹：

```text
VO 终点误差 = 0.1105
回环修正后终点误差 = 0.0
```

结果文件：

```text
labs/3.4-slam/02-loop-closure/results/loop_closure_demo.png
```

解释：

- VO 因为累计误差没有回到起点；
- 回环约束认为终点应该回到起点；
- 简化修正把误差沿路径分配；
- 真实 SLAM 使用位姿图优化或 BA 进行全局修正。

---

## 13. 本实验最重要的结论

```text
特征匹配 ≠ 相机运动
相机运动 ≠ 完整 SLAM
三角化 ≠ 完整三维地图
一次位姿优化 ≠ 完整 BA
回环修正 ≠ 完整回环检测
```

但本次实验已经把：

```text
图片
→ 特征匹配
→ 相对位姿
→ 相机轨迹
→ 三维点
→ 重投影误差
→ 简化 BA
→ 回环修正
```

完整串起来了。
