import matplotlib
from ultralytics import YOLO


def get_bb(results):
    res = results[0]
    bb = []
    bounding_boxes = res.boxes.xyxy.numpy().tolist()
    pad = 3
    extended_boxes = []
    for box in bounding_boxes:
        extended_box = [box[0] - pad, box[1] - pad, box[2] + pad, box[3] + pad]
        extended_boxes.append(extended_box)
    for cls, conf, pts in zip(
        res.boxes.cls.numpy().tolist(),
        res.boxes.conf.numpy().tolist(),
        extended_boxes,
    ):
        bb.append([res.names[cls], conf, pts])
    return bb


def get_bot(bb):
    x, y = 0, 1
    bbd = dict(query=[], matrix=[], trains=[])
    for cls, conf, pts in bb:
        is_pt_bot = pts[y] > 200
        match cls:
            case "0":
                if is_pt_bot:
                    bbd["query"].append(pts)
                else:
                    bbd["matrix"].append(pts)
            case "1":
                bbd["trains"].append(pts)
    bbd["query"].sort(key=lambda i: i[x])
    bbd["trains"].sort(key=lambda i: i[x])
    horizontal = sorted(bbd["matrix"], key=lambda x: x[0], reverse=False)
    vertical = sorted(bbd["matrix"], key=lambda x: x[1], reverse=True)
    bbd["matrix_aligned"] = vertical[:-3] + horizontal[3:]
    return bbd


def get_pairs_coords(anchor, rest):
    return [[anchor, i] for i in rest]


def normalize_train_boxes(data):
    frame_coords = [0, 200, 400, 600, 800, 1000, 1200]
    for i, j in zip(data, frame_coords):
        i[0] -= j
        i[2] -= j
    return data


def run(model_path, image_path):
    model = YOLO(model_path)
    results = model.predict(image_path, save=False)
    bb = get_bb(results)
    bd = get_bot(bb)
    l = get_pairs_coords(
        anchor=bd["query"][0],
        rest=bd["matrix_aligned"],
    )
    r = get_pairs_coords(
        anchor=bd["query"][1],
        rest=bd["matrix_aligned"],
    )
    t = normalize_train_boxes(bd["trains"])
    return [l, r, t]


if __name__ == "__main__":
    matplotlib.use("tkagg")
    run(
        model_path="./MODELS/best.pt",
        image_path="../DATA/yolo_short/images/test/2645.jpg",
    )
    print("##DONE")
