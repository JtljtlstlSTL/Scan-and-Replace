import cv2
import numpy as np
import os
import subprocess
from multiprocessing import Pool

# 设置路径
input_video = 'input_video.mp4'  # 输入视频文件
target_face_image = 'target_face.png'  # 目标人脸图像
frame_folder = 'frames'  # 存储提取的帧
output_folder = 'processed_frames'  # 存储处理后的帧
output_video = 'output_video.mp4'  # 输出视频文件

# 创建文件夹
os.makedirs(frame_folder, exist_ok=True)
os.makedirs(output_folder, exist_ok=True)

# 步骤 1：使用 FFmpeg 提取视频帧
def extract_frames():
    print("正在提取视频帧...")
    try:
        subprocess.run(
            ['ffmpeg', '-i', input_video, '-vf', 'fps=30', os.path.join(frame_folder, 'frame_%04d.png')],
            check=True
        )
        print("视频帧提取完成！")
    except subprocess.CalledProcessError as e:
        print(f"提取视频帧时出错: {e}")
        exit(1)

# 步骤 2：人脸检测（使用 OpenCV DNN）
def detect_faces(image):
    # 加载模型
    model_file = "deploy.prototxt"
    weight_file = "res10_300x300_ssd_iter_140000.caffemodel"
    net = cv2.dnn.readNetFromCaffe(model_file, weight_file)

    # 预处理图像
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

    # 输入网络并获取检测结果
    net.setInput(blob)
    detections = net.forward()

    faces = []
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:  # 置信度阈值
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            faces.append((startX, startY, endX - startX, endY - startY))

    return faces

# 步骤 3：对齐人脸
def align_face(image, face):
    x, y, w, h = face
    return image[y:y+h, x:x+w]  # 直接裁剪人脸区域

# 步骤 4：替换人脸
def replace_face(frame, face, target_face):
    x, y, w, h = face
    # 调整目标人脸大小以匹配检测到的人脸
    target_face_resized = cv2.resize(target_face, (w, h))
    # 将目标人脸覆盖到原图上
    frame[y:y+h, x:x+w] = target_face_resized
    return frame

# 步骤 5：处理单帧
def process_frame(frame_name):
    frame_path = os.path.join(frame_folder, frame_name)
    image = cv2.imread(frame_path)

    # 如果图像读取失败，跳过该帧
    if image is None:
        print(f"无法读取帧: {frame_name}")
        return

    # 加载目标人脸图像
    target_face = cv2.imread(target_face_image)
    if target_face is None:
        print(f"无法读取目标人脸图像: {target_face_image}")
        return

    # 人脸检测
    faces = detect_faces(image)

    # 如果没有检测到人脸，直接保存原图
    if len(faces) == 0:
        output_path = os.path.join(output_folder, frame_name)
        cv2.imwrite(output_path, image)
        print(f"未检测到人脸，保存原图: {frame_name}")
        return

    # 替换人脸
    for face in faces:
        aligned_face = align_face(image, face)
        if aligned_face is None:
            continue

        # 替换人脸
        image = replace_face(image, face, target_face)

    # 保存处理后的帧
    output_path = os.path.join(output_folder, frame_name)
    cv2.imwrite(output_path, image)
    print(f"处理并保存帧: {frame_name}")

# 步骤 6：并行处理所有帧
def process_frames():
    print("正在处理视频帧...")
    frame_names = [f for f in os.listdir(frame_folder) if f.endswith('.png')]
    with Pool(processes=4) as pool:  # 使用4个进程并行处理
        pool.map(process_frame, frame_names)
    print("视频帧处理完成！")

# 步骤 7：使用 FFmpeg 将处理后的帧合成为视频
def create_video():
    print("正在合成视频...")
    try:
        subprocess.run(
            [
                'ffmpeg', '-r', '30', '-i', os.path.join(output_folder, 'frame_%04d.png'),
                '-c:v', 'libx264', '-vf', 'fps=30', '-pix_fmt', 'yuv420p', output_video
            ],
            check=True
        )
        print("视频合成完成！")
    except subprocess.CalledProcessError as e:
        print(f"合成视频时出错: {e}")
        exit(1)

# 主函数
def main():
    print("开始人脸替换流程...")

    # 提取视频帧
    extract_frames()

    # 处理视频帧
    process_frames()

    # 合成为视频
    create_video()

    print("人脸替换完成！输出视频：", output_video)

if __name__ == "__main__":
    main()