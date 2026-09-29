#回环检测和轨迹修正
# 相机绕一圈回到起点
#   → VO 估计的终点没有回到起点
#   → 回环检测发现“这里以前来过”
#   → 加入回环约束
#   → 后端修正轨迹和地图

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

#当前脚本目录
base_dir=Path(__file__).parent
result_dir=base_dir/"results"
result_dir.mkdir(exist_ok=True)


#第一部分：创建一条“真实的绕圈路线”
def make_true_loop():
    """
    创建一个闭合的矩形轨迹。

    相机路径：
    左下 -> 右下 -> 右上 -> 左上 -> 回到左下
    """

    #相机在不同时间的位置
    path=[]

    #下边：左->右 y固定为0
    #np.linspace(0.0,4.0,41)意思：在 0 到 4 之间均匀取 41 个数字
    for x in np.linspace(0.0,4.0,41):
        path.append([x,0.0])

    #右边：下->上 x固定为4
    #np.linspace(0.0,3.0,31)意思：在 0 到 3 之间均匀取 31 个数字
    #[1:] 是 Python 的切片操作。意思是：从第 1 个元素开始，取到最后一个，所以 [1:] 会去掉第一个 0.0
    #因为这个点已经加入过了，上面
    for y in np.linspace(0.0,3.0,31)[1:]:
        path.append([4.0,y])

    #上边：右->左 y固定为3
    #np.linspace(4.0,0.0,41)意思：在 4 到 0 之间均匀取 41 个数字
    #[1:] 是 Python 的切片操作。意思是：从第 1 个元素开始，取到最后一个，所以 [1:] 会去掉第一个 4.0
    #因为这个点已经加入过了，上面
    for x in np.linspace(4.0,0.0,41)[1:]:
        path.append([x,3.0])

    #左边：上->下 回到起点 x固定为0
    #np.linspace(3.0,0.0,31)意思：在 3 到 0 之间均匀取 31 个数字
    #[1:] 是 Python 的切片操作。意思是：从第 1 个元素开始，取到最后一个，所以 [1:] 会去掉第一个 3.0
    #因为这个点已经加入过了，上面
    for y in np.linspace(3.0,0.0,31)[1:]:
        path.append([0.0,y])

    return np.asarray(path,dtype=float)


#第二个函数：故意制造一个“有漂移的 VO”
def simulate_vo(true_path,step_scale_error=0.03,noise_std=0.005,seed=0):
    """
    模拟 VO 的累计漂移。

    每一步都在真实运动基础上加入：
    1. 统一的尺度误差
    2. 少量随机噪声
    """
    rng=np.random.default_rng(seed)

    # 真实轨迹的第一个点就是：[0,0]
    # 所以一开始让 VO 也从：[0,0]开始。
    # 也就是说：一开始两者完全一样。
    estimated_path=[true_path[0].copy()]

    #然后一个位置一个位置地模拟
    for i in range(1,len(true_path)):
        #计算真实的一小步
        true_delta=true_path[i]-true_path[i-1]

        #尺度误差会造成累积漂移
        estimated_delta=true_delta.copy()

        #故意给这一小步增加 3% 的错误
        estimated_delta *= (1.0 + step_scale_error)

        #加一点随机误差 即像把原来的数据给小改一下随机
        estimated_delta += rng.normal(
            0.0,
            noise_std,
            size=2,
        )

        estimated_path.append(
            #把这一小步加到之前的位置
            #当前位置 = 上一个估计位置 + 这一步估计移动量
            #estimated_path[-1] 是最后一个元素，即上一个元素（因为上一个是加进来的刚刚）
            estimated_path[-1]+estimated_delta
        )

    return np.asarray(estimated_path,dtype=float)



#第三个函数：模拟“回环修正”
def apply_loop_closure(vo_path):
    """
    使用一个简化方法模拟回环修正。

    真实 SLAM 会做位姿图优化。
    这里为了让轨迹形状看得清楚，采用：
    起点 - 终点 的误差，沿路径逐渐分配。
    """
    # 这个代码做了一个非常简单的回环约束
    # correction=corrected[0]-corrected[-1]
    # 假设：起点：[0,0]  VO终点：[0.5, -0.2]
    # 那么：correction=[0,0] - [0.5,-0.2]=[-0.5,0.2]
    # 什么意思？就是：终点需要往左 0.5，再往上 0.2。
    corrected=vo_path.copy()

    #回环约束认为：终点应该回到起点
    correction=corrected[0]-corrected[-1]

    for i in range(len(corrected)):
        t=i/(len(corrected)-1)
        corrected[i,:2]+=t*correction

    return corrected



#真实轨迹
true_path=make_true_loop()

#模拟vo轨迹
vo_path=simulate_vo(true_path)

#回环修正后的轨迹
corrected_path=apply_loop_closure(vo_path)

#计算终点误差
vo_end_error=np.linalg.norm(
    vo_path[-1]-true_path[-1]
)

corrected_end_error=np.linalg.norm(
    corrected_path[-1]-true_path[-1]
)

print("真实终点:",np.round(true_path[-1],4))
print("vo 终点:",np.round(vo_path[-1],4))

print("vo 终点误差:",round(float(vo_end_error),4))
print("修正后终点误差:",round(float(corrected_end_error),4))

#绘制轨迹
fig,ax=plt.subplots(figsize=(8,6))

ax.plot(
    true_path[:,0],
    true_path[:,1],
    color="green",
    linewidth=3,
    label="true path"
)

ax.plot(
    vo_path[:,0],
    vo_path[:,1],
    color="red",
    linewidth=2,
    label="VO path with drift"
)

ax.plot(
    corrected_path[:,0],
    corrected_path[:,1],
    color="blue",
    linestyle="--",
    linewidth=2,
    label="loop-closure corrected"
)

#标出起点和终点
ax.scatter(
    true_path[0,0],
    true_path[0,1],
    color="black",
    s=80,
    label="start"
)

ax.scatter(
    vo_path[-1,0],
    vo_path[-1,1],
    color="orange",
    s=80,
    label="VO end"
)

ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_title("VO drift and loop-closure correction")
ax.axis("equal")
ax.grid(True)
ax.legend()

plot_path = result_dir / "loop_closure_demo.png"
plt.savefig(plot_path, dpi=200, bbox_inches="tight")

print("回环演示图已保存:", plot_path)

plt.close(fig)




