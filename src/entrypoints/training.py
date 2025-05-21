import pprint
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import loguniform
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR

from src.entrypoints.helpers import load_cub_dataset, load_satellites_dataset  # noqa: F401
from src.fine_tuning.adapter import add_adapters_to_model_layers
from src.fine_tuning.methods import layernorm_tuning, lora_all_linear, lora_attn  # noqa: F401
from src.fine_tuning.trainer import ModelTrainer
from src.models.vitgpt2 import ViTGPT2


def get_random_params(seed: int | None = None) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    return {
        "max_lr": float(loguniform.rvs(1e-5, 1e-2)),
        "div_factor": float(rng.uniform(25, 250)),
        "weight_decay": float(rng.choice([0, 1e-4, 1e-3])),
        "batch_size": int(rng.choice([2, 4])),
    }
    

def random_search(max_epochs: int = 10, num_samples: int = 10) -> list[dict[str, Any]]:
    run_data: list[dict[str, Any]] = []

    for _ in range(num_samples):
        run_configs = get_random_params()
        print(run_configs)
        run_configs["epochs"] = max_epochs
        final_val_metrics = train_model(run_configs)

        run_data.append({
            "run_config": run_configs,
            "validation_metrics": final_val_metrics,
        })
    return run_data


def train_model(config: dict[str, Any]) -> dict[str, list[float]]:
    epochs = config["epochs"]
    batch_size = config["batch_size"]
    training_subset, validation_subset = load_satellites_dataset(batch_size)
    eps = 1e-6 if config["max_lr"] > 5e-4 else 1e-8  # noqa: PLR2004

    optimizer_configs = {
        "lr": 1e-4,
        "betas": (0.9, 0.985),
        "eps": eps,
        "weight_decay": config["weight_decay"],
    }

    scheduler_configs = {
        "max_lr": config["max_lr"],
        "epochs": epochs,
        "steps_per_epoch": len(training_subset),
        "div_factor": config["div_factor"],
    }

    model = ViTGPT2("cuda:0")
    add_adapters_to_model_layers(model, 11, 64)
    optimizer = Adam(model.model.parameters(), **optimizer_configs) # type: ignore
    scheduler = OneCycleLR(optimizer, **scheduler_configs) # type: ignore

    trainer = ModelTrainer(model, optimizer, scheduler)
    trainer.train(epochs, training_subset, validation_subset)
    trainer.save_training_config(epochs, batch_size, optimizer_configs, scheduler_configs)
    return trainer.get_validation_metrics()


def main() -> None:
    results = random_search(25, 30)
    results.sort(key=lambda x: x["validation_metrics"]["loss"])
    best_result = results[0]
    
    pprint.pprint(best_result)  # noqa: T203
    with Path.open(Path("./test_results.txt"), "w") as test_results:
        test_results.write(pprint.pformat(results))


if __name__ == "__main__":
    main()
