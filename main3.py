from ultralytics import YOLO
import pandas as pd


model = YOLO('yolov8n.pt')                    # transfer learning
model.train(
    data = "start.yaml",
    epochs  = 100,
    classes = [3, 4],                       # 3 = smoke, 4 = fire
    patience=20,
    lr0=0.003,
    hsv_h=0.015,        # as fire, smoke have differing saturation, hue settings
    hsv_s=0.7,
    hsv_v=0.4,
    imgsz=960,
    per_class=True
)

# reading the best values for each metric
# as it didnt differ between 2 classes I've uploaded a common metrics
df = pd.read_csv(r"C:\Users\nurpe\fs\pythonProject\runs\detect\train27\results.csv")

last = df.iloc[-1]

print(f"Precision: {last['metrics/precision(B)']:.3f}")
print(f"Recall:    {last['metrics/recall(B)']:.3f}")
print(f"mAP50:     {last['metrics/mAP50(B)']:.3f}")
