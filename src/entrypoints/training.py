from typing import Any

from ray import train, tune
from ray.train import Checkpoint, get_checkpoint
from ray.tune.schedulers import ASHAScheduler
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR

from src.entrypoints.helpers import load_cub_dataset
from src.fine_tuning.adapter import add_adapters_to_model_layers  # noqa: F401
from src.fine_tuning.methods import layernorm_tuning, lora_all_linear, lora_attn  # noqa: F401
from src.fine_tuning.trainer import ModelTrainer
from src.models.vitgpt2 import ViTGPT2


def ray_tune(max_epochs: int = 25, num_samples: int = 10) -> None:
    search_space = {
        "max_lr": tune.loguniform(1e-5, 1e-2),
        "div_factor": tune.uniform(25, 250),
        "weight_decay": tune.loguniform(0, 1e-4),
        "batch_size": tune.choice([2, 4, 6]),
        "epochs": max_epochs,
    }
    
    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=max_epochs,
        grace_period=2,
        reduction_factor=2,
    )
    
    result = tune.run(
        train_model,
        resources_per_trial={"cpu": 16, "gpu": 1},
        config=search_space,
        num_samples=num_samples,
        scheduler=scheduler,
    )
    
    best_trial = result.get_best_trial("loss", "min", "last")
    if best_trial is None:
        msg = "All trials returned NaN."
        raise ValueError(msg)

    print(f"Best trial config: {best_trial.config}")  # noqa: T201
    print(f"Best trial final validation loss: {best_trial.last_result['loss']}")  # noqa: T201
    print(f"Best trial final validation accuracy: {best_trial.last_result['accuracy']}")  # noqa: T201


def train_model(config: dict[str, Any]) -> None:
    epochs = config["epochs"]
    batch_size = config["batch_size"]
    training_subset, validation_subset = load_cub_dataset(batch_size)

    optimizer_configs = {
        "lr": 1e-4,
        "betas": (0.9, 0.985),
        "eps": 1e-8,
        "weight_decay": config["weight_decay"],
    }

    scheduler_configs = {
        "max_lr": config["max_lr"],
        "epochs": epochs,
        "steps_per_epoch": len(training_subset),
        "div_factor": config["div_factor"],
    }

    model = ViTGPT2("cuda:0")
    lora_all_linear(model)
    optimizer = Adam(model.model.parameters(), **optimizer_configs) # type: ignore
    scheduler = OneCycleLR(optimizer, **scheduler_configs) # type: ignore

    trainer = ModelTrainer(model, optimizer, scheduler)
    trainer.train(epochs, training_subset, validation_subset)
    trainer.save_training_config(epochs, batch_size, optimizer_configs, scheduler_configs)


def main() -> None:
    ray_tune(10, 5)


if __name__ == "__main__":
    main()
