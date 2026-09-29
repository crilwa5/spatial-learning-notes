from pathlib import Path

import cv2




# 第 1 步 从视频中每隔几帧提取一张图片，形成连续图像序列
#当前脚本所在目录
base_dir=Path(__file__).parent

#视频路径
video_path=base_dir/"video"/"walk.MP4"

#输出图片序列目录
sequence_dir=base_dir/"sequence"
sequence_dir.mkdir(exist_ok=True)


#定义缩放函数
def resize_keep_ratio(image,max_side=1280):
    """
    按比例缩小图片，使最长边不超过 max_side。
    """
    heigth,width=image.shape[:2]
    scale=min(1.0,max_side/max(heigth,width))

    if scale<1.0:
        new_width=int(width*scale)
        new_height=int(heigth*scale)

        return cv2.resize(
            image,
            (new_width,new_height),
            interpolation=cv2.INTER_AREA
        )

    return image


#打开视频
cap=cv2.VideoCapture(str(video_path))  #打开视频文件

if not cap.isOpened():
    raise FileNotFoundError(f"视频打开失败:{video_path}")

#读取视频基本信息
fps=cap.get(cv2.CAP_PROP_FPS)  #帧率，每秒多少帧
frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  #总帧数
width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  #分辨率
heigth=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) #分辨率

#时长 = 总帧数 / 帧率
duration=frame_count/fps if fps>0 else 0.0  #视频时长，单位秒

print("视频路径:",video_path)
print("FPS:",fps)
print("总帧数:",frame_count)
print("分辨率:",width,"x",heigth)
print("时长(秒):",round(duration,2))

#每隔多少帧取一张
frame_step=15

#最多保存多少张，避免一次性生成太多
max_frames=30

frame_index=0  #当前读取到第几帧
saved_count=0  #已经保存了多少张

#循环读取视频帧
while True:
    # success：是否成功
    # frame：当前帧图像
    success,frame=cap.read()  #读取一帧

    if not success:
        break

    #每隔frame_step帧保存一次
    if frame_index % frame_step == 0:
        frame=resize_keep_ratio(frame)  #把当前帧按比例缩小

        output_path=sequence_dir/ f"{saved_count}.jpg"  #保存路径：sequence/0.jpg、sequence/1.jpg、……

        #避免中文路径问题，使用imencode + tofile 保存
        success_encode,encoded=cv2.imencode(  #把图片编码成字节
            ".jpg",
            frame,
            [cv2.IMWRITE_JPEG_QUALITY,95]  #JPEG 质量 95
        )

        if not success_encode:
            raise RuntimeError(f"图片编码失败:{output_path}")

        encoded.tofile(str(output_path))  #写入文件

        saved_count+=1

        if saved_count>=max_frames:
            break

    frame_index+=1

cap.release()  #释放视频

print("抽取图片数量:",saved_count)
print("图片保存位置:",sequence_dir)






# 第 2 步 连续帧之间的 VO 轨迹累积
#1）把前面“两帧特征匹配 → R、t”的流程封装成函数，并先用序列中的前两帧测试
# vo_trajectory.py里