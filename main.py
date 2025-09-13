import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# --------------------------
# 1. Load YOLOv10 model
# --------------------------
model = YOLO("yolov10s.pt")   # or your fine-tuned model

# --------------------------
# 2. Initialize DeepSORT
# --------------------------
tracker = DeepSort(
    max_age=30,
    n_init=3,
    max_cosine_distance=0.3,
    nn_budget=None
)

# --------------------------
# 3. Open video / webcam
# --------------------------
video_path = "/workspaces/CodeAlpha_Object-detection-and-tracking/3.mp4"   # Change to 0 for webcam
cap = cv2.VideoCapture(video_path)

# Get video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Define video writer (output file)
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("output.mp4", fourcc, fps if fps > 0 else 30, (width, height))

# --------------------------
# 4. Process frames
# --------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Step 1: Run YOLOv10 detection
    results = model.predict(frame, conf=0.5, verbose=False)

    detections = []
    for r in results[0].boxes:
        x1, y1, x2, y2 = r.xyxy[0].tolist()
        conf = float(r.conf[0])
        cls = int(r.cls[0])
        detections.append(([x1, y1, x2 - x1, y2 - y1], conf, cls))
        # Format: [x, y, w, h], confidence, class_id

    # Step 2: Update DeepSORT tracker
    tracks = tracker.update_tracks(detections, frame=frame)

    # Step 3: Draw results
    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = track.track_id
        ltrb = track.to_ltrb()  # left, top, right, bottom
        x1, y1, x2, y2 = map(int, ltrb)

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw class name + track ID
        label = f"{model.names[cls]} ID {track_id}" if cls in model.names else f"ID {track_id}"
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Write frame to output video
    out.write(frame)

# --------------------------
# 5. Clean up
# --------------------------
cap.release()
out.release()
print("✅ Tracking complete. Output saved as 'output.mp4'")
