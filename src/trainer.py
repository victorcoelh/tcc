import torch
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from tqdm import tqdm

from basemodel import Model
from src.dataset import Dataset


class ModelTrainer:
    def __init__(self, model: Model) -> None:
        self.__model = model.model
        self.__processor = model.processor
        self.__tokenizer = model.tokenizer
        
        self.__optimizer = AdamW(
            self.__model.parameters(),
            lr=0.001,
            betas=(0.9, 0.999),
            weight_decay=0.01,
        )
        
        self.__loss = CrossEntropyLoss()
        
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
            
            for i, (images, captions) in enumerate(pbar):
                print(end="\r", flush=True)  # noqa: T201
                
                captions_tensor = self.__tokenizer(captions, padding=True)
                captions_tensor = torch.Tensor(captions_tensor["input_ids"]).cuda().long()
                
                logits, loss = self.__model_predict(images, captions_tensor)
                total_loss += loss.mean().item()
                
                loss = self.__loss(logits, captions_tensor)
                
                pbar.set_description(f"Epoch {epoch+1}/{epochs}, "
                                     f"Training Loss: {(total_loss/i+1):.4f}")
                
                loss.backward()
                self.__optimizer.step()
                torch.cuda.empty_cache()
                
            avg_training_loss = total_loss / len(training_dataset)
            avg_validation_loss, val_cider = self.__evaluate(validation_dataset)
            
            self.__save_metrics(avg_training_loss, avg_validation_loss)
            
    def __model_predict(self, images: ImageBatch, captions: torch.Tensor,
                        ) -> tuple[torch.Tensor, torch.Tensor]:
        pass
    
    def __evaluate(self, dataset: Dataset) -> tuple[torch.Tensor, float]:
        pass
    
    def __save_metrics(self):
        pass

    def freeze_model_layers(self, layers_to_train: list[str]) -> None:
        for layer_name, param in self.__model.named_parameters():
            if not any(name in layer_name for name in layers_to_train):
                param.requires_grad = False
