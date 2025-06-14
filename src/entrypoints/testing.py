from pathlib import Path

import numpy as np
import pandas as pd
import safetensors.torch
import torch
from peft import PeftModel

from src.data_loading import data_loading
from src.data_loading.dataset import Dataset
from src.fine_tuning.adapter import add_adapters_to_model_layers  # noqa: F401
from src.fine_tuning.methods import wise
from src.models.basemodel import Model
from src.models.vitgpt2 import ViTGPT2
from src.utils.metrics import (
    calculate_bertscore,
    calculate_cider,
    calculate_meteor,
    calculate_spice,
)


def test_model(model: Model, dataset: Dataset) -> pd.DataFrame:
    cider_values = []
    meteor_values = []
    bertscore_values = []
    spice_values = []
    
    for images, captions in dataset:
        batch_size = len(images)
        predictions = model.predict(images, batch_size)

        cider_values.extend(calculate_cider(captions, predictions, batch_size))
        meteor_values.extend(calculate_meteor(captions, predictions, batch_size))
        bertscore_values.extend(calculate_bertscore(captions, predictions, batch_size))
        spice_values.extend(calculate_spice(captions, predictions, batch_size))

    cider_values = pd.Series(cider_values).describe()
    meteor_values = pd.Series(meteor_values).describe()
    bertscore_values = pd.Series(bertscore_values).describe()
    spice_values = pd.Series(spice_values).describe()

    return pd.DataFrame([cider_values, meteor_values, bertscore_values, spice_values])


def wise_grid_search(model_path: str, dataset: Dataset) -> None:
    base_model = ViTGPT2("cuda:0")
    
    for i in np.linspace(0.8, 1.0, 21):
        ft_model = ViTGPT2("cuda:0", endpoint=model_path)
        wise(ft_model, base_model, i)
        outcome = test_model(ft_model, dataset)
        with Path.open(Path("wise_results.txt"), "a") as f:
            f.write(f"{i}: {outcome["mean"].to_numpy()}\n\n")
            

def load_cub_dataset(batch_size: int) -> Dataset:
    dataset_dir = Path("dataset/cub200/")
    ids_path = Path("dataset/cub200/train_test_split.txt")

    paths, captions = data_loading.from_cub(dataset_dir, ids_path, training=False)
    return Dataset(paths, captions, batch_size)


def load_satellites_dataset(batch_size: int) -> Dataset:
    images_dir = "dataset/satellite/"
    test_csv = "dataset/satellite/test.csv"
    
    paths, captions = data_loading.from_csv(test_csv, images_dir)
    return Dataset(paths, captions, batch_size, None)


def load_safetensors(file_path: str) -> dict[str, torch.Tensor]:
    with Path.open(Path(file_path), "rb") as f:
        binaries = f.read()
        return safetensors.torch.load(binaries)


def main() -> None:
    testing_subset = load_satellites_dataset(128)
    
    base_model = ViTGPT2("cuda:0")
    model = ViTGPT2("cuda:0", endpoint="runs/updated/satellites/naive/best_cider/")
    
    wise(model, base_model, 0.91)
    
    outcome = test_model(model, testing_subset)
    print(outcome)


if __name__ == "__main__":
    main()
