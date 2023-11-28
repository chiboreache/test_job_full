import random
import shutil
from pathlib import Path


def copy_file_with_folder_creation(src, dst, ext):
    if not dst.exists():
        dst.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst / ext)


def dataset_split(ls, ratio, rnd):
    if rnd:
        random.shuffle(ls)
    test = ls[: ratio["test"]]
    remaining_items = ls[ratio["test"] :]
    split_index = int(len(remaining_items) * ratio["train"] / 100)
    train = remaining_items[:split_index]
    val = remaining_items[split_index:]
    return {"test": test, "train": train, "val": val}


def run():
    src = Path("../DATA/dataset_yolo_full")
    dst = Path("../DATA/yolo_full")
    img_path = dst / "images"
    lbl_path = dst / "labels"
    name_list = [i.stem for i in src.glob("*.jpg")]
    names = dataset_split(ls=name_list, ratio=dict(test=100, train=80), rnd=True)
    for dataset_type, ls in names.items():
        print(f"dataset_type: {dataset_type}")
        for i in ls:
            jpg = f"{i}.jpg"
            txt = f"{i}.txt"
            copy_file_with_folder_creation(src / jpg, img_path / dataset_type, jpg)
            copy_file_with_folder_creation(src / txt, lbl_path / dataset_type, txt)


if __name__ == "__main__":
    run()
    print("##DONE")
