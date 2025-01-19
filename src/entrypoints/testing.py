import pandas as pd

from src.data_loading.dataset import Dataset, get_path_and_captions
from src.models.basemodel import Model
from src.models.vitgpt2 import ViTGPT2
from src.utils.metrics import calculate_cider, calculate_meteor


def test_model(model: Model, dataset: Dataset) -> pd.DataFrame:
    cider_values = []
    meteor_values = []
    
    for images, captions in dataset:
        batch_size = len(images)
        
        predictions = model.predict(images, 16)
        cider_values.append(calculate_cider(captions, predictions, batch_size))
        meteor_values.append(calculate_meteor(captions, predictions, batch_size))
        
    cider_values = pd.Series(cider_values).describe()
    meteor_values = pd.Series(meteor_values).describe()
        
    return pd.DataFrame([cider_values, meteor_values])
    

def main() -> None:
    images_dir = "dataset/satellite/"
    test_csv = "dataset/satellite/test.csv"

    test_paths, test_captions = get_path_and_captions(test_csv, images_dir)
    testing_subset = Dataset(test_paths, test_captions, 4, None)
    
    model = ViTGPT2("cuda:0", endpoint="runs/test_run/")
    
    outcome = test_model(model, testing_subset)
    print(outcome)  # noqa: T201


if __name__ == "__main__":
    main()
