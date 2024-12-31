from abc import ABC, abstractmethod

from transformers.image_processing_utils import BaseImageProcessor
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils import PreTrainedTokenizer

from src.type_hints import ImageBatch


class Model(ABC):
    @property
    @abstractmethod
    def model(self) -> PreTrainedModel:
        pass

    @property
    @abstractmethod
    def processor(self) -> BaseImageProcessor:
        pass

    @property
    @abstractmethod
    def tokenizer(self) -> PreTrainedTokenizer:
        pass

    @abstractmethod
    def predict(self, images: ImageBatch, batch_size: int) -> list[str]:
        pass
