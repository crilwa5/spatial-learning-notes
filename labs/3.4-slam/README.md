# 3.4 SLAM 最小实验

这个目录不是完整的工业级 SLAM，而是一条用于理解视觉 SLAM 基本流程的最小实验链：

```text
视频
  → 抽帧
  → ORB 特征匹配
  → RANSAC 筛选
  → 基础矩阵 F
  → 本质矩阵 E
  → recoverPose
  → 相对 R、t
  → 累积相机轨迹
  → 三角化三维点
  → 重投影误差
  → solvePnP 位姿优化
  → 简化交替 BA
  → 回环修正
```

当前实验的目的：

- 理解 VO 和 SLAM 的核心数据流；
- 理解相机位姿、深度、三维点和重投影误差的关系；
- 理解后端优化为什么存在；
- 为后续三维场景理解、场景图和 3D 空间问答打基础；
- 不要求现在实现完整 ORB-SLAM、VIO 或工业级后端。

---

## 目录

```text
labs/3.4-slam/
├─ 01-feature-matching/
│  ├─ images/
│  │  ├─ 0001.JPG
│  │  └─ 0002.JPG
│  ├─ video/
│  │  └─ walk.MP4
│  ├─ sequence/
│  │  ├─ 0.jpg
│  │  └─ ...
│  ├─ read_images.py
│  ├─ extract_frames.py
│  ├─ vo_trajectory.py
│  └─ results/
└─ 02-loop-closure/
   ├─ loop_closure_demo.py
   └─ results/
```

注意：

- `video/walk.MP4` 和 `sequence/` 是输入数据，通常不建议提交到 Git；
- `results/` 里只保留少量小图片和文本结果；
- 代码和笔记需要提交，大视频和大量中间图片可以只保留在本地。

---

## 运行环境

本次实验使用独立环境 `slam_env`：

```text
Python 3.10
opencv-python 5.0.0
numpy 2.2.6
matplotlib 3.10.9
```

对应 `requirements.txt` 中应包含：

```text
numpy
opencv-python
matplotlib
```

---

## 运行顺序

### 1. 读取两张图片

```powershell
python labs/3.4-slam/01-feature-matching/read_images.py
```

完成内容：

- 读取 `0001.JPG` 和 `0002.JPG`；
- 按比例缩小到最长边 1280；
- 转成灰度图；
- 检测 ORB 特征点。

### 2. 从视频抽帧

```powershell
python labs/3.4-slam/01-feature-matching/extract_frames.py
```

完成内容：

- 读取 `walk.MP4`；
- 每 15 帧保存一张图片；
- 保存到 `sequence/`。

### 3. 简化 VO、三角化和 BA

```powershell
python labs/3.4-slam/01-feature-matching/vo_trajectory.py
```

完成内容：

- ORB 特征匹配；
- RANSAC 筛选；
- 估计相对 R、t；
- 累积相机轨迹；
- 三角化三维点；
- 计算重投影误差；
- solvePnP 位姿优化；
- 简化交替 BA；
- 保存轨迹、点云、误差和图片。

### 4. 回环修正演示

```powershell
python labs/3.4-slam/02-loop-closure/loop_closure_demo.py
```

完成内容：

- 模拟一条闭合轨迹；
- 加入累计漂移；
- 用简化回环约束修正终点误差；
- 保存轨迹对比图。

---

## 重要说明

### 1. 相机内参是近似的

本实验没有对手机相机正式标定，使用的是近似内参：

```text
fx = fy = max(width, height)
cx = width / 2
cy = height / 2
```

所以：

- 轨迹形状和相对关系可以参考；
- 不能把坐标直接当作真实米数；
- 真实项目应该先做相机标定。

### 2. 单目视觉存在尺度不确定性

`recoverPose` 得到的 `t` 只有方向：

```text
t_length = 1
```

因此整条轨迹只有归一化尺度。

恢复真实尺度需要：

- IMU；
- RGB-D；
- 已知相机基线；
- 已知物体真实尺寸；
- 外部定位系统。

### 3. 简化 BA 不是完整 BA

本实验采用交替优化：

```text
固定三维点，用 solvePnP 优化相机位姿
固定相机位姿，重新三角化三维点
重复若干轮
```

真正的 BA 会同时联合优化：

- 相机位姿；
- 三维点；
- 有时还包括相机内参和畸变参数。

### 4. 实验边界

本实验没有实现：

- 完整 ORB-SLAM；
- 视觉惯性 VIO；
- 闭环误检验证；
- 位姿图优化；
- 全局地图管理；
- 工业级鲁棒性处理。

---

## 后续衔接

这个实验连接到：

```text
相机位姿
  → 深度 / 点云
  → 三维地图
  → 三维场景理解
  → 场景图
  → 3D-LLM / 空间问答
```

对于三维空间智能问答方向，重点不是把 SLAM 继续做到工业级，而是理解：

> 相机位姿、深度、三维点和地图是怎样产生并为后续空间理解提供几何证据的。
