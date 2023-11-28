import json
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

plt.style.use("dark_background")


def run():
    src = "../DATA"
    ds_path = f"{src}/dataset"
    num = 2756
    json_path = f"{ds_path}/{num}.json"
    with open(str(json_path), "r") as file:
        data = json.load(file)
        image_height = data["imageHeight"]
        image_width = data["imageWidth"]
        image_path = f"{ds_path}/{data['imagePath']}"
        image = plt.imread(image_path)
        fig, ax = plt.subplots(figsize=(21, 8))
        ax.imshow(image)
        for shape in data["shapes"]:
            label = shape["label"]
            points = np.array(shape["points"])
            x_values = points[:, 0]
            y_values = points[:, 1]
            if shape["shape_type"] == "point":
                ax.plot(x_values, y_values, "ro", label=label)
            elif shape["shape_type"] == "rectangle":
                x_min, y_min = np.min(x_values), np.min(y_values)
                width = np.max(x_values) - x_min
                height = np.max(y_values) - y_min
                rect = patches.Rectangle(
                    (x_min, y_min),
                    width,
                    height,
                    linewidth=1,
                    edgecolor="r",
                    facecolor="none",
                    label=label,
                )
                ax.add_patch(rect)
                ax.text(
                    x_min,
                    y_min - 5,
                    label,
                    color="white",
                    fontsize=8,
                    bbox=dict(facecolor="black", alpha=0.7),
                )
        plt.title("Visualizing Shapes on Image")
        plt.gca().set_facecolor("black")
        plt.tight_layout()
        plt.show()
        plt.close()


if __name__ == "__main__":
    run()
    print("##DONE")
