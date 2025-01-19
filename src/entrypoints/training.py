from torch.optim import AdamW
from torch.optim.lr_scheduler import ConstantLR

from src.data_loading.dataset import Dataset, get_path_and_captions
from src.fine_tuning.trainer import ModelTrainer
from src.models.vitgpt2 import ViTGPT2


def main() -> None:
    images_dir = "dataset/satellite/"
    train_csv = "dataset/satellite/train.csv"
    valid_csv = "dataset/satellite/valid.csv"
    
    epochs = 15
    batch_size = 5
    
    training_paths, training_captions = get_path_and_captions(train_csv, images_dir)
    training_subset = Dataset(training_paths, training_captions, batch_size, None)
    
    validation_paths, validation_captions = get_path_and_captions(valid_csv, images_dir)
    validation_subset = Dataset(validation_paths, validation_captions, 16, None)
    
    optimizer_configs = {
        "lr": 1e-6,
        "betas": (0.9, 0.98),
        "eps": 1e-6,
        "weight_decay": 0.01,
    }
    
    scheduler_configs = {
        "factor": 1e-2,
        "total_iters": 1000,
    }
    
    model = ViTGPT2("cuda:0")
    optimizer = AdamW(model.model.parameters(), **optimizer_configs) # type: ignore
    scheduler = ConstantLR(optimizer, **scheduler_configs) # type: ignore
    
    trainer = ModelTrainer(model, optimizer, scheduler)
    trainer.train(epochs, training_subset, validation_subset)
    trainer.save_training_config(epochs, batch_size, optimizer_configs, scheduler_configs)


if __name__ == "__main__":
    main()
