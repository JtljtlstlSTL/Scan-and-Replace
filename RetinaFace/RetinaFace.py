#使用 RetinaFace 进行人脸检测和关键点定位
from retinaface import RetinaFace
import cv2
import numpy as np

def detect_faces(image):
    """
    使用 RetinaFace 检测人脸和关键点
    """
    # 检测人脸
    faces = RetinaFace.detect_faces(image)
    if isinstance(faces, dict):
        return faces
    else:
        return None

def main():
    # 1. 加载输入图像
    image = cv2.imread('input_image.jpg')
    if image is None:
        print("Error: 无法加载图像")
        return

    # 2. 使用 RetinaFace 进行人脸检测和关键点定位
    faces = detect_faces(image)
    if faces is None:
        print("未检测到人脸")
        return

    # 3. 绘制检测结果
    for face_id, face_info in faces.items():
        # 获取人脸框和关键点
        facial_area = face_info['facial_area']
        landmarks = face_info['landmarks']

        # 绘制人脸框
        x, y, w, h = facial_area
        cv2.rectangle(image, (x, y), (w, h), (0, 255, 0), 2)

        # 绘制关键点
        for landmark_name, landmark_point in landmarks.items():
            cv2.circle(image, (int(landmark_point[0]), int(landmark_point[1])), 2, (0, 0, 255), -1)

    # 4. 显示结果
    cv2.imshow("Detected Faces", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
