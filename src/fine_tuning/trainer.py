import torch
from matplotlib import pyplot as plt
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from tqdm import tqdm

from src.data_loading.dataset import Dataset
from src.models.basemodel import Model
from src.utils.type_hints import ImageBatch


class ModelTrainer:
    def __init__(self, model: Model) -> None:
        self.__model = model.model
        self.__processor = model.processor
        self.__tokenizer = model.tokenizer
        
        self.__loss = CrossEntropyLoss()
        self.__optimizer = AdamW(
            self.__model.parameters(),
            lr=0.001,
            betas=(0.9, 0.999),
            weight_decay=0.01,
        )
        
        self.training_loss = []
        self.validation_loss = []
        self.training_cider = []
        self.validation_cider = []

    def train(self, epochs: int, training_dataset: Dataset, validation_dataset: Dataset) -> None:
        self.__model.train()
        
        for epoch in range(epochs):
            self.__model.train()
            pbar = tqdm(training_dataset, total=len(training_dataset))
            total_loss = 0
            total_cider = 0
            
            for i, (images, captions) in enumerate(pbar):
                print(end="\r", flush=True)  # noqa: T201
                
                captions_tensor = self.__tokenizer(captions, padding=True)
                captions_tensor = torch.Tensor(captions_tensor["input_ids"]).cuda().long()
                
                logits, loss = self.__model_predict(images, captions_tensor)
                total_loss += loss.mean().item()
                
                pbar.set_description(f"Epoch {epoch+1}/{epochs}, "
                                     f"Training Loss: {(total_loss/(i+1)):.4f}")
                
                loss.backward()
                self.__optimizer.step()
                torch.cuda.empty_cache()
                
            training_loss = total_loss / len(training_dataset)
            training_cider = total_cider / len(training_dataset)
            validation_loss, validation_cider = self.__evaluate(validation_dataset)
            
            self.__save_metrics(training_loss, validation_loss, training_cider, validation_cider)
            
    def __model_predict(self, images: ImageBatch,
                        captions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.autocast(device_type="cuda"):
            image_tensor = self.__processor(images=images, return_tensors="pt")
            image_tensor = torch.Tensor(image_tensor["pixel_values"]).cuda()
            
            outputs = self.__model(pixel_values=image_tensor, labels=captions)
            logits = outputs.logits
            loss = self.__loss(logits.transpose(1, 2), captions)

        return logits, loss
    
    def __calculate_cider(self, _logits: torch.Tensor, _caption: torch.Tensor) -> float:
        return 0
    
    def __evaluate(self, dataset: Dataset) -> tuple[float, float]:
        self.__model.eval()
        total_loss = 0
        total_cider = 0
        
        with torch.no_grad():
            for (images, captions) in dataset:
                captions_tensor = self.__tokenizer(captions, padding=True)
                captions_tensor = torch.Tensor(captions_tensor["input_ids"]).cuda().long()
                
                logits, loss = self.__model_predict(images, captions_tensor)
                total_loss += loss.mean().item()
                
        avg_loss = total_loss / len(dataset)
        avg_cider = total_cider / len(dataset)
        return avg_loss, avg_cider
    
    def __save_metrics(self, training_loss: float, validation_loss: float,
                       training_cider: float, validation_cider: float) -> None:
        self.training_loss.append(training_loss)
        self.validation_loss.append(validation_loss)
        self.training_cider.append(training_cider)
        self.validation_cider.append(validation_cider)
        
        epochs = len(self.training_loss)
        
        plt.figure(figsize=(12, 8))
        plt.subplot(1, 2, 1)
        plt.plot(epochs, training_loss)
        plt.plot(epochs, validation_loss)
        
        plt.subplot(1, 2, 2)
        plt.plot(epochs, training_cider)
        plt.plot(epochs, validation_cider)
        
        plt.savefig("training_metrics.png")

    def freeze_model_layers(self, layers_to_train: list[str]) -> None:
        for layer_name, param in self.__model.named_parameters():
            if not any(name in layer_name for name in layers_to_train):
                param.requires_grad = False
