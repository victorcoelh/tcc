from datetime import datetime
from pathlib import Path
from typing import Any

import nltk
import numpy as np
import torch
from matplotlib import pyplot as plt
from nltk.translate import meteor_score
from pycocoevalcap.cider.cider import Cider
from torch.nn import CrossEntropyLoss
from tqdm import tqdm

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
        
        self.training_loss = []
        self.validation_loss = []
        
        self.training_cider = []
        self.validation_cider = []
        
        self.training_meteor = []
        self.validation_meteor = []
        
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
        
        for epoch in range(epochs):
            self.__model.train()
            pbar = tqdm(training_dataset, total=len(training_dataset))
            total_loss = 0
            total_cider = 0
            total_meteor = 0
            
            for i, (images, captions) in enumerate(pbar):
                torch.cuda.empty_cache()
                print(end="\r", flush=True)  # noqa: T201
                batch_size = len(images)
                
                single_captions = [np.random.default_rng().choice(caption)
                                   for caption in captions]

                captions_tensor = self.__tokenizer(single_captions, padding=True)
                captions_tensor = torch.Tensor(captions_tensor["input_ids"]).cuda().long()
                
                logits, loss = self.__model_predict(images, captions_tensor)
                total_loss += loss.mean().item()
                total_cider += self.__calculate_cider(logits, captions, batch_size)
                total_meteor += self.__calculate_meteor(logits, captions, batch_size)
                
                pbar.set_description(f"Epoch {epoch+1}/{epochs}, "
                                     f"Training Loss: {(total_loss/(i+1)):.4f} ")
                
                loss.backward()
                self.__optimizer.step()
                self.__scheduler.step()
                
            training_loss = total_loss / len(training_dataset)
            training_cider = total_cider / len(training_dataset)
            training_meteor = total_meteor / len(training_dataset)

            validation_loss, validation_cider, validation_meteor =\
                self.__evaluate(validation_dataset)
                
            print(f"Validation Meteor: {validation_meteor:.2f}, Cider: {validation_cider:.2f}")  # noqa: T201
            
            self.__save_metrics(training_loss, validation_loss,
                                training_cider, validation_cider,
                                training_meteor, validation_meteor)

        self.__save_model()
        
    def save_training_config(self, epochs: int, batch_size: int, optimizer_config: dict[str, Any],
                             scheduler_config: dict[str, Any]) -> None:
        with Path.open(Path(f"./runs/{self.time}/config.txt"), "w") as f:
            f.write(f"Epochs: {epochs}\n")
            f.write(f"Batch Size: {batch_size}\n")
            f.write(f"Optimizer: {type(self.__optimizer)}\n")
            f.write(f"Scheduler: {type(self.__scheduler)}\n")
            f.write(f"Optimizer Config: {optimizer_config}\n")
            f.write(f"Scheduler Config: {scheduler_config}\n")
            
    def __model_predict(self, images: ImageBatch,
                        captions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        image_tensor = self.__processor(images=images, return_tensors="pt")
        image_tensor = torch.Tensor(image_tensor["pixel_values"]).cuda()
        
        with torch.autocast(device_type="cuda"):
            outputs = self.__model(pixel_values=image_tensor, labels=captions)
            logits = outputs.logits
            loss = self.__loss(logits.transpose(1, 2), captions)

        return logits, loss
    
    def __calculate_cider(self, logits: torch.Tensor, target: list[list[str]],
                          batch_size: int) -> float:
        token_ids = torch.argmax(logits, dim=-1).cpu().numpy().reshape(batch_size, -1)
        reference: dict[int, list[str]] = {}
        predicted: dict[int, list[str]] = {}
        
        for i in range(batch_size):
            ref_captions = self.__tokenizer(target[i])["input_ids"]
            ref_captions = [" ".join(str(token) for token in ref)
                            for ref in ref_captions] # type: ignore

            reference[i] = ref_captions
            predicted[i] = [" ".join(str(token) for token in token_ids[i])]
            
        return Cider(n=4, sigma=6.0).compute_score(reference, predicted)[0] # type: ignore
    
    def __calculate_meteor(self, logits: torch.Tensor, target: list[list[str]],
                           batch_size: int) -> float:
        token_ids = torch.argmax(logits, dim=-1).cpu().numpy().reshape(batch_size, -1)
        score = 0
        
        for i in range(batch_size):
            predicted_caption = self.__tokenizer.decode(token_ids[i])
            references = target[i]
            
            reference_tokens = [nltk.word_tokenize(ref) for ref in references]
            generated_tokens = nltk.word_tokenize(predicted_caption)
            
            score += meteor_score.meteor_score(reference_tokens, generated_tokens)
            
        return score / batch_size
    
    def __evaluate(self, dataset: Dataset) -> tuple[float, float, float]:
        self.__model.eval()
        total_loss = 0
        total_cider = 0
        total_meteor = 0
        
        with torch.no_grad():
            for (images, captions) in dataset:
                batch_size = len(images)
                
                single_captions = [caption[0] for caption in captions]
                captions_tensor = self.__tokenizer(single_captions, padding=True)
                captions_tensor = torch.Tensor(captions_tensor["input_ids"]).cuda().long()
                
                logits, loss = self.__model_predict(images, captions_tensor)
                total_loss += loss.mean().item()
                total_cider += self.__calculate_cider(logits, captions, batch_size)
                total_meteor += self.__calculate_meteor(logits, captions, batch_size)
                
        avg_loss = total_loss / len(dataset)
        avg_cider = total_cider / len(dataset)
        avg_meteor = total_meteor / len(dataset)
        return avg_loss, avg_cider, avg_meteor
    
    def __save_metrics(self, training_loss: float, validation_loss: float,  # noqa: PLR0913
                       training_cider: float, validation_cider: float,
                       training_meteor: float, validation_meteor: float) -> None:
        self.training_loss.append(training_loss)
        self.validation_loss.append(validation_loss)
        
        self.training_cider.append(training_cider)
        self.validation_cider.append(validation_cider)

        self.training_meteor.append(training_meteor)
        self.validation_meteor.append(validation_meteor)
        
        epochs = list(range(1, len(self.training_loss)+1))
        
        plt.figure(figsize=(12, 8))
        plt.subplot(3, 1, 1)
        plt.plot(epochs, self.training_loss, "bo-", label="Training Loss")
        plt.plot(epochs, self.validation_loss, "ro-", label="Validation Loss")
        plt.legend()
        plt.grid()
        
        plt.subplot(3, 1, 2)
        plt.plot(epochs, self.training_cider, "bo-", label="Training Cider")
        plt.plot(epochs, self.validation_cider, "ro-", label="Validation Cider")
        plt.legend()
        plt.grid()
        
        plt.subplot(3, 1, 3)
        plt.plot(epochs, self.training_meteor, "bo-", label="Training Meteor")
        plt.plot(epochs, self.validation_meteor, "ro-", label="Validation Meteor")
        plt.legend()
        plt.grid()
        
        plt.tight_layout()
        plt.savefig(f"./runs/{self.time}/training_metrics.png")
        
    def __save_model(self) -> None:
        self.__model.save_pretrained(f"./runs/{self.time}")
        self.__processor.save_pretrained(f"./runs/{self.time}")
        self.__tokenizer.save_pretrained(f"./runs/{self.time}")
