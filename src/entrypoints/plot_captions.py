from matplotlib import pyplot as plt

from src.entrypoints.testing import load_cub_dataset, load_satellites_dataset, load_safetensors
from src.fine_tuning.adapter import add_adapters_to_model_layers
from src.fine_tuning.methods import wise
from src.models.vitgpt2 import ViTGPT2
from src.utils.type_hints import Image


def plot_image(img: Image) -> None:
    plt.imshow(img)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig("example.png")


def main() -> None:
    dataset = load_cub_dataset(1)
    print(f"Image Path: {dataset.image_paths[0]}\n\n")

    img_test, captions = dataset[0]
    print(f"Ground Truth: {captions[:3]}")

if __name__ == "__main__":
    main()
