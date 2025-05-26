import cv2
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO('best.pt')
names = model.names


# Open video
cap = cv2.VideoCapture("people.mp4")
# Debug mouse position
def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        print(f"Mouse moved to: [{x}, {y}]")

cv2.namedWindow("RGB")
cv2.setMouseCallback("RGB", RGB)


line_y=304

frame_count = 0
hs={}
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % 3 != 0:
        continue

    frame = cv2.resize(frame, (1020,600))
    results = model.track(frame, persist=True)

    if results[0].boxes.id is not None:
        ids = results[0].boxes.id.cpu().numpy().astype(int)
        boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
        class_ids = results[0].boxes.cls.int().cpu().tolist()

        for track_id, box, class_id in zip(ids, boxes, class_ids):
            x1, y1, x2, y2 = box
            label = names[class_id]
            cx=int(x1+x2)//2
            cy=int(y1+y2)//2
            if track_id in hs:
                prev_cx,prev_cy=hs[track_id]
                if prev_cy<line_y<=cy:
                  cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                  cv2.putText(frame, label, (x1 + 3, y1 - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                  cv2.putText(frame, str(track_id), (x2 + 3, y2 - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            hs[track_id]=(cx,cy)
          

           
    cv2.line(frame,(0,line_y),(frame.shape[1],line_y),(0,0,255),2)
    cv2.imshow("RGB", frame)
    if cv2.waitKey(0) & 0xFF == 27:  # ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
