import json
import shutil
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("tkagg")
plt.style.use("dark_background")
plt.figure(figsize=(10, 10))


def cls_mapping(lbl):
    m = {
        "1": 0,
        "2": 1,
        "3": 2,
        "4": 3,
        "5": 4,
        "6": 5,
        "14": 6,
        "15": 7,
        "16": 8,
        "24": 9,
        "25": 10,
        "26": 11,
        "34": 12,
        "35": 13,
        "36": 14,
    }
    return m[lbl]


def to_yolo(label, pt, h, w):
    x1, y1 = pt[0]
    x2, y2 = pt[1]
    x_center = (x1 + x2) / (2 * w)
    y_center = (y1 + y2) / (2 * h)
    box_width = abs(x2 - x1) / w
    box_height = abs(y2 - y1) / h
    return f"{cls_mapping(label)} {x_center:.6f} {y_center:.6f} {box_width:.6f} {box_height:.6f}\n"


def proc_ds(data):
    yolo = ""
    w = data["imageWidth"]
    h = data["imageHeight"]
    for shape in data["shapes"]:
        if shape["shape_type"] == "rectangle":
            label = shape["label"]
            if not label.isdigit():
                label = label.replace("_", "")
            yolo += to_yolo(label, shape["points"], h, w)
    return yolo


def write_yolo(dst, txt_name, yolo):
    with open(dst / txt_name, "w") as output_file:
        output_file.write(yolo)


def dir_to_path(dst_path, src_path):
    src = Path(src_path)
    dst = Path(dst_path)
    dst.mkdir(parents=True, exist_ok=True)
    return dst, src


def run(
    src_path="../DATA/dataset",
    dst_path="dataset_yolo_full",
):
    dst, src = dir_to_path(dst_path=dst_path, src_path=src_path)
    for json_path in src.glob("*.json"):
        with open(json_path, "r") as f:
            print(f"json_path: {json_path}")
            data = json.load(f)
            yolo = proc_ds(data)
            jpg_name = f"{json_path.stem}.jpg"
            txt_name = f"{json_path.stem}.txt"
            write_yolo(dst, txt_name, yolo)
            shutil.copyfile(src / jpg_name, dst / jpg_name)


if __name__ == "__main__":
    run()
    print("##DONE")
