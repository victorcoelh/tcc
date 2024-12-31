import torch
import transformers
from transformers import AutoTokenizer, VisionEncoderDecoderModel, ViTImageProcessor

from src.models.basemodel import Model
from src.utils.type_hints import ImageBatch


class ViTGPT2(Model):
    def __init__(self, device: str, endpoint: str = "nlpconnect/vit-gpt2-image-captioning") -> None:
        self.__device = device
        self.__processor = ViTImageProcessor.from_pretrained(endpoint)
        self.__tokenizer = AutoTokenizer.from_pretrained(endpoint)

        self.__model = VisionEncoderDecoderModel.from_pretrained(endpoint)
        self.__model.to(self.__device) # type: ignore
        self.__model.eval()

    def predict(self, images: ImageBatch, batch_size: int = 16) -> list[str]:
        output = []

        for i in range(0, len(images), batch_size):
            end = i + batch_size
            batch = images[i:end]
            output.extend(self.__model_predict(batch))

        return output

    def __model_predict(self, input_array: ImageBatch) -> list[list[str]]:
        predictions = []

        with torch.no_grad():
            for image in input_array:
                inputs = self.__processor(images=image, return_tensors="pt").to(self.__device)
                pixel_values = inputs["pixel_values"]

                generated_ids = self.__model.generate(
                    pixel_values,
                    max_new_tokens=150,
                    pad_token_id=self.__tokenizer.eos_token_id,
                )

                generated_text = self.__tokenizer\
                    .batch_decode(generated_ids, skip_special_tokens=True)[0]\
                    .strip()

                predictions.append(generated_text)

        return predictions

    @property
    def processor(self) -> ViTImageProcessor:
        return self.__processor

    @property
    def model(self) -> VisionEncoderDecoderModel:
        return self.__model # type: ignore

    @property
    def tokenizer(self) -> transformers.tokenization_utils.PreTrainedTokenizerBase:
        return self.__tokenizer # type: ignore
