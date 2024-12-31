from src.data_loading.dataset import Dataset, get_path_and_captions
from src.fine_tuning.trainer import ModelTrainer
from src.models.vitgpt2 import ViTGPT2


def main() -> None:
    images_dir = "dataset/satellite/"
    train_csv = "dataset/satellite/train.csv"
    valid_csv = "dataset/satellite/valid.csv"

    training_paths, training_captions = get_path_and_captions(train_csv, images_dir)
    training_subset = Dataset(training_paths, training_captions, 9, None)
    
    validation_paths, validation_captions = get_path_and_captions(valid_csv, images_dir)
    validation_subset = Dataset(validation_paths, validation_captions, 9, None)
    
    model = ViTGPT2("cuda:0")
    trainer = ModelTrainer(model)
    trainer.train(10, training_subset, validation_subset)


if __name__ == "__main__":
    main()
