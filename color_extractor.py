import cv2
import matplotlib.pyplot as plt
import numpy as np
import PIL
from sklearn.cluster import KMeans


def read(path):
    img = PIL.Image.open(path)
    h, w = img.size
    if h * w > 1e5:
        img.thumbnail((225, 225))
    return np.array(img)


def color_based_segmentation(image, num_clusters=10, hue_factor=0.5):
    assert image.dtype == np.uint8
    image = image.astype(np.float32) / 255
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    # trigonomeric embedding of hue
    pixels = np.stack(
        [
            hue_factor * np.cos(hsv[:, :, 0] / 180 * np.pi) / 8**0.5,
            hue_factor * np.sin(hsv[:, :, 0] / 180 * np.pi) / 8**0.5,
            hsv[:, :, 1],
            hsv[:, :, 2],
        ],
        axis=-1,
    ).reshape(-1, 4)
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(pixels)
    labels = kmeans.labels_

    # Create a mask for each cluster label
    masks = [(labels == i).reshape(image.shape[:2]) for i in range(num_clusters)]
    contours = []
    h, w = image.shape[:2]
    for mask in masks:
        cnts, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        for cnt in cnts:
            cnt_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(cnt_mask, [cnt], -1, 1, thickness=cv2.FILLED)
            cnt_mask = cnt_mask != 0
            contours.append((cnt, cnt_mask))

    gaussian = np.exp(
        -50
        * (
            (np.arange(h)[:, None] / h - 1 / 2) ** 2
            + (np.arange(w)[None, :] / w - 1 / 2) ** 2
        )
    )
    best_contour = max(contours, key=lambda c: np.sum(gaussian[c[1]]))
    segmented_image = np.ones_like(image)

    for contours, mask in sorted(contours, key=lambda c: cv2.contourArea(c[0])):
        # weird result for raspberries with HSV
        # color = cv2.mean(hsv, mask.astype(np.uint8))[:3]
        # color = cv2.cvtColor(np.array([[color]], dtype=np.float32), cv2.COLOR_HSV2RGB)[
        #     0, 0
        # ]
        color = cv2.mean(image, mask.astype(np.uint8))[:3]
        segmented_image[mask] = color
    best_color = cv2.mean(image, best_contour[1].astype(np.uint8))[:3]
    segmented_image[best_contour[1]] = best_color
    cv2.drawContours(segmented_image, [best_contour[0]], -1, (0, 1, 0), 1)

    return best_color, segmented_image


if __name__ == "__main__":
    import glob
    import json

    from joblib import Parallel, delayed
    from tqdm_joblib import tqdm_joblib

    paths = [path for path in glob.glob("imgs/*.jpeg") if "segmented" not in path]

    def process(path):
        img = read(path)
        best_color, segmented_image = color_based_segmentation(img)
        plt.imsave(path.replace(".jpeg", " segmented.jpeg"), segmented_image)
        return path.split("/")[1], best_color

    with tqdm_joblib(total=len(paths)) as progress_bar:
        colors = dict(Parallel(n_jobs=-1)(delayed(process)(path) for path in paths))

    with open("results.json", "r") as f:
        results = json.load(f)
    for items in results.values():
        for it in items:
            it.append(colors[it[0]])
    with open("results.json", "w") as f:
        json.dump(results, f)
