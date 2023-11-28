import json
import random
import string
from collections import Counter
from pathlib import Path
from PIL import Image


def hash():
    letters = string.ascii_lowercase
    random_letters = "".join(random.choice(letters) for _ in range(5))
    return random_letters


def expand_roi(x1, y1, x2, y2, ROI_expand=3):
    x1 = x1 - ROI_expand
    y1 = y1 - ROI_expand
    x2 = x2 + ROI_expand
    y2 = y2 + ROI_expand
    return [x1, y1, x2, y2]


def fx(xyxy):
    x1, y1 = xyxy[0]
    x2, y2 = xyxy[1]
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return x1, y1, x2, y2


def crop(image_path, coords, output_path):
    x1, y1, x2, y2 = fx(coords)
    x1, y1, x2, y2 = expand_roi(x1, y1, x2, y2)
    img = Image.open(image_path)
    cropped_img = img.crop((x1, y1, x2, y2))
    cropped_img.save(output_path)


def filter_pairs(json_data):
    rectangles = [i for i in json_data["shapes"] if i["shape_type"] == "rectangle"]
    labels = [i["label"] for i in rectangles if i["label"].isnumeric()]
    pairs = Counter(labels).most_common()
    res = []
    for q in [i[0] for i in pairs if i[1] == 2]:
        res.append([i["points"] for i in rectangles if i["label"] in q])
    return res


def run(src="../DATA/dataset", dst="res-3"):
    dst = Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    src = Path(src)
    ls = list(src.glob("*.json"))
    ls.sort()
    for json_path in ls:
        print(f"json_path: {json_path}")
        with open(json_path, "r") as f:
            data = json.load(f)
            for template, qq in enumerate(filter_pairs(data)):
                for pair, xyxy in enumerate(qq):
                    image_name = src / data["imagePath"]
                    dr = dst / f"{json_path.stem}_{template+1:02}"
                    dr.mkdir(parents=True, exist_ok=True)
                    crop(image_name, xyxy, dr / f"{pair+1:02}_{hash()}.jpg")


if __name__ == "__main__":
    run()
    print("##DONE")
