import cv2

from ultralytics import solutions

cap = cv2.VideoCapture("car01.mp4")
assert cap.isOpened(), "Error reading video file"

# Pass region as list
# region_points = [(20, 400), (1080, 400), (1080, 360), (20, 360)]

# Pass region as dictionary
region_points = {
    "region-01": [(500, 500), (800, 500), (800, 600), (500, 600)],
    "region-02": [(1000, 500), (1200, 500), (1200, 600), (1000, 600)],
}

# Video writer
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))
video_writer = cv2.VideoWriter("region_counting.avi", cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

# Initialize region counter object
regioncounter = solutions.RegionCounter(
    show=True,  # display the frame
    region=region_points,  # pass region points
    model="yolo11n.pt",  # model for counting in regions i.e yolo11s.pt
)

# Process video
while cap.isOpened():
    success, im0 = cap.read()

    if not success:
        print("Video frame is empty or processing is complete.")
        break

    results = regioncounter(im0)

    # print(results)  # access the output

    video_writer.write(results.plot_im)

cap.release()
video_writer.release()
cv2.destroyAllWindows()  # destroy all opened windows