from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

imagens = [
    "CAMINHO_PERCORRIDO_GRS.png",
    "CAMINHO_PERCORRIDO_HILLCLIMBING.png",
    "CAMINHO_PERCORRIDO_LRS.png",
    "CAMINHO_PERCORRIDO_TEMPERA.png",
]

for i in range(8):
    folder = f"out/tc3/f_{i + 1}/"
    fig, axs = plt.subplots(2, 2, tight_layout=True)
    for ax in axs.flat:
        ax.axis("off")
    for i, ax in enumerate(axs.flat):
        img = Image.open(folder + imagens[i])
        ax.imshow(np.array(img))
    plt.savefig(folder + "MERGED_RST.png", dpi=300, bbox_inches="tight")
    plt.clf()
    plt.close()
