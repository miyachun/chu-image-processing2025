from flask import Flask, Response, render_template
import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

app = Flask(__name__)


model = YOLO("yolo11n.pt")

# 初始化 DeepSORT 追蹤器
tracker = DeepSort(max_age=30)

# 嘗試開啟攝影機
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("無法開啟攝影機，請確認裝置是否存在。")

def gen_frames():
    while True:
        success, frame = cap.read()
        if not success:
            print("無法讀取攝影機影像")
            break

        # YOLO 偵測
        results = model(frame)[0]

        detections = []
        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])

            if cls_id != 0:  # 只處理 person 類別
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w, h = x2 - x1, y2 - y1
            detections.append(([x1, y1, w, h], conf, "person"))

        # 使用 DeepSORT 追蹤器
        tracks = tracker.update_tracks(detections, frame=frame)

        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            x1, y1, x2, y2 = map(int, track.to_ltrb())
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f'ID {track_id}', (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # 編碼成 JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        frame_bytes = buffer.tobytes()

        # 串流輸出
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
    
if __name__ == '__main__':
    app.run(debug=True)



@app.teardown_appcontext
def cleanup(exception=None):
    if cap.isOpened():
        cap.release()

if __name__ == '__main__':
    app.run(debug=True)
