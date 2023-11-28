from ultralytics import YOLO


model = YOLO("yolov8x.pt")

model.train(
    data="train_yolo.yaml",
    epochs=100,
    patience=5,
    imgsz=1400,
    batch=8,
    rect=True,
    degrees=0,
    scale=0.1,
    shear=0.1,
    translate=0.1,
    fliplr=0,
    copy_paste=0,
    lr0=0.001,
    lrf=0.001,
)
