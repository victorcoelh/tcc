from pathlib import Path

from sklearn.model_selection import train_test_split
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR

from src.data_loading import data_loading
from src.data_loading.dataset import Dataset
from src.fine_tuning.adapter import add_adapters_to_model_layers  # noqa: F401
from src.fine_tuning.methods import layernorm_tuning, lora_all_linear, lora_attn  # noqa: F401
from src.fine_tuning.trainer import ModelTrainer
from src.models.vitgpt2 import ViTGPT2


def load_satellites_dataset(batch_size: int) -> tuple[Dataset, Dataset]:
    images_dir = "dataset/satellite/"
    train_csv = "dataset/satellite/train.csv"
    valid_csv = "dataset/satellite/valid.csv"
    
    training_paths, training_captions = data_loading.from_csv(train_csv, images_dir)
    training_subset = Dataset(training_paths, training_captions, batch_size, None)
    
    validation_paths, validation_captions = data_loading.from_csv(valid_csv, images_dir)
    validation_subset = Dataset(validation_paths, validation_captions, 16, None)
    
    return training_subset, validation_subset
    

def load_cub_dataset(batch_size: int) -> tuple[Dataset, Dataset]:
    dataset_dir = Path("dataset/cub200/")
    ids_path = Path("dataset/cub200/train_test_split.txt")
    
    paths, captions = data_loading.from_cub(dataset_dir, ids_path, training=True)
    train_paths, valid_paths, train_captions, valid_captions = train_test_split(
        paths,
        captions,
        test_size=0.1,
        shuffle=True,
    )
    
    return (
        Dataset(train_paths, train_captions, batch_size),
        Dataset(valid_paths, valid_captions, batch_size),
    )


def main() -> None:
    epochs = 60
    batch_size = 6
    training_subset, validation_subset = load_cub_dataset(batch_size)

    optimizer_configs = {
        "lr": 1e-4,
        "betas": (0.9, 0.985),
        "eps": 1e-8,
        "weight_decay": 0,
    }

    scheduler_configs = {
        "max_lr": 1e-4,
        "epochs": epochs,
        "steps_per_epoch": len(training_subset),
        "div_factor": 100,
    }

    model = ViTGPT2("cuda:0")
    lora_all_linear(model)
    optimizer = Adam(model.model.parameters(), **optimizer_configs) # type: ignore
    scheduler = OneCycleLR(optimizer, **scheduler_configs) # type: ignore

    trainer = ModelTrainer(model, optimizer, scheduler)
    trainable_params = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.model.parameters())
    print(f"Trainable Params: {trainable_params} | Total Params: {total_params} | "  # noqa: T201
          f"% = {trainable_params*100 / total_params:.2f}")

    trainer.train(epochs, training_subset, validation_subset)
    trainer.save_training_config(epochs, batch_size, optimizer_configs, scheduler_configs)


if __name__ == "__main__":
    main()
