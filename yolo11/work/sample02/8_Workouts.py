import cv2
from ultralytics import YOLO
import os


model = YOLO("yolo11n-pose.pt")

# 開啟影片
video_path = "people02.mp4" 
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise IOError(f"無法開啟影片: {video_path}")

# 取得影片資訊
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = cap.get(cv2.CAP_PROP_FPS)

#  建立輸出影片寫入器（AVI 格式）
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 'XVID' 表示 AVI 格式
out = cv2.VideoWriter('output.avi', fourcc, fps, (width, height))


while True:
    ret, frame = cap.read()
    if not ret:
        break  # 到影片結尾

    # 執行肢體偵測
    results = model(frame, verbose=False)
    result = results[0]

    # 畫出關鍵點
    annotated_frame = result.plot()

    # 寫入輸出影片
    out.write(annotated_frame)


cap.release()
out.release()
cv2.destroyAllWindows()

print("輸出已儲存為 output.avi")
