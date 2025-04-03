import pickle
import tempfile
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from ray import train
from ray.train import Checkpoint, get_checkpoint
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from transformers.modeling_utils import PreTrainedModel

from src.data_loading.dataset import Dataset
from src.models.basemodel import Model
from src.utils.type_hints import ImageBatch


class ModelTrainer:
    def __init__(self, model: Model, optimizer: torch.optim.Optimizer,
                 scheduler: torch.optim.lr_scheduler.LRScheduler) -> None:
        self.__model = model.model
        self.__processor = model.processor
        self.__tokenizer = model.tokenizer

        self.__loss = CrossEntropyLoss()
        self.__optimizer = optimizer
        self.__scheduler = scheduler
        
        time = datetime.now(None)   # noqa: DTZ005
        self.time = time.strftime("%d-%m_%H:%M:%S")
        Path(f"./runs/{self.time}").mkdir(parents=True)

    def freeze_model_layers(self, layers_to_train: list[str]) -> None:
        for layer_name, param in self.__model.named_parameters():
            if not any(name in layer_name for name in layers_to_train):
                param.requires_grad = False

    def print_model_layers(self) -> None:
        for layer_name, _ in self.__model.named_parameters():
            print(layer_name)  # noqa: T201

    def train(self, epochs: int, training_dataset: Dataset, validation_dataset: Dataset) -> None:
        self.__model.train()
        start_epoch = load_checkpoint(self.__model, self.__optimizer) # type: ignore
        
        for epoch in range(start_epoch, epochs):
            pbar = tqdm(training_dataset, total=len(training_dataset))
            self.__model.train()
            total_loss = 0
            
            for i, (images, captions) in enumerate(pbar):
                torch.cuda.empty_cache()
                print(end="\r", flush=True)  # noqa: T201
                
                single_captions = [np.random.default_rng().choice(caption)
                                   for caption in captions]

                captions_tensor = self.__tokenizer(single_captions, padding=True)
                captions_tensor = torch.Tensor(captions_tensor["input_ids"]).cuda().long()

                logits, loss = self.__model_predict(images, captions_tensor)
                total_loss += loss.mean().item()
                
                pbar.set_description(f"Epoch {epoch+1}/{epochs}, "
                                     f"Training Loss: {(total_loss/(i+1)):.4f} ")
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.__model.parameters(), 1)
                self.__optimizer.step()
                self.__scheduler.step()
                
            training_loss = total_loss / len(training_dataset)
            validation_loss = self.__evaluate(validation_dataset)
            
            checkpoint_data = {
                "epoch": epoch,
                "net_state_dict": self.__model.state_dict(),
                "optimizer_state_dict": self.__optimizer.state_dict(),
            }
            
            with tempfile.TemporaryDirectory() as checkpoint_dir:
                data_path = Path(checkpoint_dir) / "data.pkl"
                with Path.open(data_path, "wb") as fp:
                    pickle.dump(checkpoint_data, fp)
                    
                checkpoint = Checkpoint.from_directory(checkpoint_dir)
                train.report(
                    {"loss": validation_loss, "train_loss": training_loss},
                    checkpoint=checkpoint,
                )

        print("Finished training. Saving model...")  # noqa: T201
        self.__save_model()
            
    def __model_predict(self, images: ImageBatch,
                        captions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        image_tensor = self.__processor(images=images, return_tensors="pt")
        image_tensor = torch.Tensor(image_tensor["pixel_values"]).cuda()
        
        with torch.autocast(device_type="cuda"):
            outputs = self.__model(pixel_values=image_tensor, labels=captions)
            logits = outputs.logits
            loss = self.__loss(logits.transpose(1, 2), captions)

        return logits, loss
    
    def __evaluate(self, dataset: Dataset) -> float:
        self.__model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for (images, captions) in dataset:
                single_captions = [caption[0] for caption in captions]
                captions_tensor = self.__tokenizer(single_captions, padding=True)
                captions_tensor = torch.Tensor(captions_tensor["input_ids"]).cuda().long()
                
                logits, loss = self.__model_predict(images, captions_tensor)
                total_loss += loss.mean().item()

        return total_loss / len(dataset)

    def __save_model(self) -> None:
        self.__model.save_pretrained(f"./runs/{self.time}")
        self.__processor.save_pretrained(f"./runs/{self.time}")
        self.__tokenizer.save_pretrained(f"./runs/{self.time}")


def load_checkpoint(model: PreTrainedModel, optimizer: torch.optim.Optimizer) -> int:
    checkpoint = get_checkpoint()
    if not checkpoint:
        return 0
    
    with checkpoint.as_directory() as checkpoint_dir:
        with Path.open(Path(checkpoint_dir) / "data.pkl", "rb") as fp:
            checkpoint_state = pickle.load(fp)  # noqa: S301
        model.load_state_dict(checkpoint_state["net_state_dict"])
        optimizer.load_state_dict(checkpoint_state["optimizer_state_dict"])
        return checkpoint_state["epoch"]
