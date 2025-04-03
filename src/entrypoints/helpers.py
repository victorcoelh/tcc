from pathlib import Path

from sklearn.model_selection import train_test_split

from src.data_loading import data_loading
from src.data_loading.dataset import Dataset


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
