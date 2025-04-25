import matplotlib.pyplot as plt
import numpy as np
from itertools import chain, combinations

from typing import List


def display_images_with_prediction(
    images, labels, classifier, n_cols=4, fig_size=(12, 6)
):
    plt.style.use("ggplot")
    n_images = len(images)
    n_rows = int(np.ceil(n_images / n_cols))
    plt.figure(figsize=fig_size)
    for idx in range(n_images):
        ax = plt.subplot(n_rows, n_cols, idx + 1)
        image = images[idx]
        pred = classifier.predict(image.unsqueeze(0)).item()
        image = image.detach().cpu().permute(1, 2, 0)
        cmap = "gray" if image.shape[2] == 1 else plt.cm.viridis  # type: ignore
        ax.set_title(f"{labels[idx].item()} ({pred})")
        ax.imshow(image, cmap=cmap)
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()
    plt.show()
    return


def powerset_without_null(s: List):
    return chain.from_iterable(combinations(s, r) for r in range(1, len(s) + 1))
