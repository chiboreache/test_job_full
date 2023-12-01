import io
import json
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path


def crop_xyxy(img, pt):
    x, y, xx, yy = pt
    crop = img.crop((x, y, xx, yy))
    bf = io.BytesIO()
    crop.save(bf, format="PNG")
    bf.seek(0)
    return bf


def get_min_item_in(d):
    return min(d.items(), key=lambda x: float(x[0]))


def get_nearest(img, pairs):
    d = {}
    for i, item in enumerate(pairs):
        a, b = item
        buff_a = crop_xyxy(img, a)
        buff_b = crop_xyxy(img, b)
        dist = eval_siamese.run(
            onnx_path=f"MODELS/pairs.onnx",
            t1=buff_a,
            t2=buff_b,
        )
        d[dist] = {
            "index": i,
            "matrix_id": i + 1,
            "xyxy_a": a,
            "xyxy_b": b,
            "buff_a": buff_a,
            "buff_b": buff_b,
        }
    return get_min_item_in(d)[1]


def is_it_true(frame, image_name, yaml_true_path):
    import yaml

    with open(yaml_true_path, "r") as file:
        d = yaml.safe_load(file)
        p = [i for i in d if i["image"] == image_name][0]
        return p["frame"] == frame


def process(image_path, dst):
    dst.mkdir(parents=True, exist_ok=True)
    img = Image.open(str(image_path))
    l, r, t = eval_yolo.run(
        model_path="MODELS/best.pt",
        image_path=str(image_path),
    )
    d_left = get_nearest(img, pairs=l)
    d_top = get_nearest(img, pairs=r)
    fig, ax = plt.subplots(figsize=(20, 10))
    image = mpimg.imread(str(image_path))
    ax.imshow(image)
    frame_number = eval_cv.run(
        box_left=d_left["xyxy_b"],
        box_top=d_top["xyxy_b"],
        trains=t,
        ax=ax,
    )
    matrix_ids = f'{d_left["matrix_id"]}-{d_top["matrix_id"]}'
    answer = is_it_true(
        frame=frame_number,
        image_name=image_path.name,
        yaml_true_path="DATA/valid_points.yaml",
    )
    print(f"\nmatrix_ids: {matrix_ids}")
    print(f"frame_number: {frame_number}\n")
    d = dst / str(answer)
    d.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{d}/{image_path.stem}_{matrix_ids}_{frame_number}.jpg")


def get_valid_frame(json_path):
    def find_frame(x=517.0):
        frame_coords = [0, 200, 400, 600, 800, 1000, 1200, 1400]
        index = 0
        for i in range(len(frame_coords) - 1):
            if frame_coords[i] <= x < frame_coords[i + 1]:
                index = i
                break
        frame = index + 1
        return index, frame

    with open(json_path, "r") as f:
        data = json.load(f)
        for shape in data["shapes"]:
            if shape["label"] == "+":
                true_x, _ = shape["points"][0]
                index, frame = find_frame(x=true_x)
                return frame


def run(
    src="../TEST",
    dst="../TEST_RES",
):
    src = Path(src)
    dst = Path(dst)
    dst.mkdir(parents=True, exist_ok=True)
    jsons = list(src.glob("*.json"))
    for json_path in jsons:
        valid_frame = get_valid_frame(json_path)
        image_path = src / f"{json_path.stem}.jpg"
        img = Image.open(str(image_path))
        try:
            l, r, t = eval_yolo.run(
                model_path="MODELS/best.pt",
                image_path=str(image_path),
            )
            d_left = get_nearest(img, pairs=l)
            d_top = get_nearest(img, pairs=r)
            fig, ax = plt.subplots(figsize=(20, 10))
            image = mpimg.imread(str(image_path))
            ax.imshow(image)
            frame_number = eval_cv.run(
                box_left=d_left["xyxy_b"], box_top=d_top["xyxy_b"], trains=t, ax=ax
            )
            if frame_number:
                matrix_ids = f'{d_left["matrix_id"]}-{d_top["matrix_id"]}'
                answer = valid_frame == frame_number
                print(f"\nmatrix_ids: {matrix_ids}")
                print(f"frame_number: {frame_number}\n")
                d = dst / str(answer)
                d.mkdir(parents=True, exist_ok=True)
                fig.savefig(f"{d}/{image_path.stem}_{matrix_ids}_{frame_number}.jpg")
            else:
                d = dst / "False_CV"
                d.mkdir(parents=True, exist_ok=True)
                fig.savefig(f"{d}/{image_path.stem}_failed.jpg")
        except Exception:
            d = dst / "False_YOLO"
            d.mkdir(parents=True, exist_ok=True)
            img.save(f"{d}/{image_path.stem}_failed.jpg")

    t = dst / "True"
    q = len(list(t.glob("*.jpg")))
    total = len(jsons)
    print(f"\n\n### Valid: [{q}/{total}]")
    print(f"### Quality: {q/total:.0%}\n\n")


def single_run(
    image_path="DATA/test/2645.jpg",
    dst_path="single_run",
):
    dst = Path(dst_path)
    image_src = Path(image_path)
    process(image_src, dst)


def test_run(
    image_src="DATA/test",
    dst_path="test_run",
):
    dst = Path(dst_path)
    src = Path(image_src)
    for num, image_path in enumerate(src.glob("*.jpg")):
        print(f"#: [{num}/100]")
        print(f"#: {image_path}")
        try:
            process(image_path, dst)
        except Exception:
            pass
    t = dst / "True"
    q = len(list(t.glob("*.jpg")))
    print(f"\n\nQuality: {q}%\n\n")


if __name__ == "__main__":
    import eval_siamese
    import eval_yolo
    import eval_cv

    # single_run()
    # test_run()
    run()
else:
    from . import eval_siamese, eval_yolo, eval_cv
