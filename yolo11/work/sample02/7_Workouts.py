import cv2
from ultralytics import YOLO
from datetime import datetime
import os

model = YOLO("yolo11n-pose.pt")
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    raise RuntimeError("  無法開啟攝影機，請檢查裝置或權限")


while True:
    ret, frame = cap.read()
    if not ret:
        print("  讀取影格失敗，跳過")
        continue

    # （可選）調整輸入大小以提升 FPS，例如 640x480
    # frame = cv2.resize(frame, (640, 480))

    # 3‑1 執行推論（直接丟 numpy array）
    results = model(frame, verbose=False)  # 回傳 list，這裡只有一張所以取 [0]
    result = results[0]

    # 3‑2 取得包含關鍵點標註的影格
    annotated = result.plot()  # Ultraytics 會畫骨架和信心度

    # 3‑3 顯示到螢幕
    cv2.imshow("YOLOv11 Pose ‑ Press q to quit, s to save", annotated)

    # 3‑4 處理鍵盤事件
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        # 儲存當前影格到檔案
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"result_{ts}.jpg"
        cv2.imwrite(filename, annotated)
        print(f"  已存檔 {filename} ({annotated.shape[1]}x{annotated.shape[0]})")

# 清理資源
cap.release()
cv2.destroyAllWindows()
