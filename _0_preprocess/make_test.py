import json
from pathlib import Path
import csv


def conv_csv():
    import csv
    import yaml

    csv_file = "valid_points.csv"
    yaml_file = "valid_points.yaml"

    def convert_to_int(val):
        try:
            return int(val)
        except ValueError:
            return val

    data = []
    with open(csv_file, newline="") as csvfile:
        csvreader = csv.DictReader(csvfile)
        for row in csvreader:
            converted_row = {key: convert_to_int(value) for key, value in row.items()}
            data.append(converted_row)
    with open(yaml_file, "w") as yfile:
        yaml.dump(data, yfile, sort_keys=False)


def read_csv():
    with open("valid_points.csv", mode="r") as file:
        csv_reader = csv.DictReader(file, skipinitialspace=True)
        ch = {header.strip(): header for header in csv_reader.fieldnames}
        for row in csv_reader:
            print(row[ch["image"]], row[ch["frame"]])


def gen_ls():  # [0, 200, 400, 600, 800, 1000, 1200, 1400]
    frames = 7
    width = 1400
    div_width = width / frames
    return [0] + [int(i * div_width) for i in range(1, frames)] + [width]


def dir_to_path(dst_path, src_path):
    src = Path(src_path)
    dst = Path(dst_path)
    dst.mkdir(parents=True, exist_ok=True)
    return dst, src


def find_frame(x=517.0):
    frame_coords = [0, 200, 400, 600, 800, 1000, 1200, 1400]
    index = 0
    for i in range(len(frame_coords) - 1):
        if frame_coords[i] <= x < frame_coords[i + 1]:
            index = i
            break
    frame = index + 1
    return index, frame


def proc_cv(data):
    true_pt = [0.0, 0.0]
    for shape in data["shapes"]:
        if shape["label"] == "+":
            true_pt = shape["points"][0]
    return true_pt


def run(
    img_src_path="../DATA/yolo_short/images/test",
    json_src_path="../DATA/dataset",
):
    img_src, json_src = dir_to_path(img_src_path, json_src_path)
    res = "image,x,y,index,frame\n"
    for jpg in sorted(img_src.glob("*.jpg")):
        fname = jpg.stem
        json_path = json_src / f"{fname}.json"
        with open(json_path, "r") as f:
            data = json.load(f)
            x, y = proc_cv(data)
            index, frame = find_frame(x=x)
            res += f"{fname}.jpg,{int(x)},{int(y)},{index},{frame}\n"
    with open("valid_points.csv", "w") as f:
        f.write(res)


if __name__ == "__main__":
    run()
    conv_csv()
    print("##DONE")
