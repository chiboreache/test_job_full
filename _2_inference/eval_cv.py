import matplotlib.pyplot as plt
from shapely.geometry import box, LineString


def draw_match(frame, ax):
    main_width = 1400
    main_height = 400
    num_divisions = 7
    division_width = main_width / num_divisions
    div_x_ls = [0] + [i * division_width for i in range(1, num_divisions)] + [main_width]
    fl = plt.Rectangle(
        (div_x_ls[frame - 1], 0),
        division_width,
        main_height,
        facecolor="#00fb3b30",
    )
    ax.add_patch(fl)
    [
        plt.axvline(
            x=i * division_width,
            color="#979797",
            linestyle="--",
        )
        for i in range(1, num_divisions)
    ]


def get_box_centroid_lines(bb, viz, ax):
    x, y, xx, yy = bb.bounds
    cx, cy = bb.centroid.x, bb.centroid.y
    line_len = 140
    hl = LineString([(x - line_len, cy), (xx + line_len, cy)])
    vl = LineString([(cx, y - line_len), (cx, yy + line_len)])
    if viz:
        x, y = bb.exterior.xy
        ax.plot(
            x,
            y,
            color="green",
            linewidth=2,
        )
        x, y = hl.xy
        ax.plot(
            x,
            y,
            color="red",
            linewidth=2,
        )
        x, y = vl.xy
        ax.plot(
            x,
            y,
            color="blue",
            linewidth=2,
        )
    return hl, vl


def lines_intersection(line1, line2):
    if line1.intersects(line2):
        return line1.intersection(line2)
    else:
        print("Lines do not intersect.")
        return None


def is_in_xywh(point_x, point_y, box_x, box_y, box_width, box_height):
    p1 = box_x <= point_x <= box_x + box_width
    p2 = box_y <= point_y <= box_y + box_height
    return p1 and p2


def is_in_xyxy(point_x, point_y, box_x, box_y, box_xx, box_yy):
    p1 = box_x <= point_x <= box_xx
    p2 = box_y <= point_y <= box_yy
    return p1 and p2


def get_frame_number(pt, trains):
    frame = None
    pad = 8
    for i, box in enumerate(trains):
        extended_box = [box[0] - pad, box[1] - pad, box[2] + pad, box[3] + pad]
        q = is_in_xyxy(*pt, *extended_box)
        if q:
            frame = i + 1
    return frame


def run(box_left, box_top, trains, ax):
    bb_left = box(*box_left)
    bb_top = box(*box_top)
    viz = True
    hl1, vl1 = get_box_centroid_lines(bb_left, viz, ax)
    hl2, vl2 = get_box_centroid_lines(bb_top, viz, ax)
    cp = lines_intersection(hl1, vl2)
    cross_pt = cp.x, cp.y
    frame_number = get_frame_number(cross_pt, trains)
    if viz and frame_number:
        draw_match(frame_number, ax)
        plt.plot(*cp.xy, "mx", markersize=35)
        plt.plot(*cp.xy, "m.", markersize=30)
        plt.axis("off")
    return frame_number


if __name__ == "__main__":
    pass
