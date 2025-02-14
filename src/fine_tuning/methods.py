from peft import LoraConfig  # type: ignore

from src.fine_tuning.trainer import ModelTrainer
from src.models.basemodel import Model


def layernorm_tuning(trainer: ModelTrainer) -> None:
    trainer.freeze_model_layers([
        "layernorm",
        #"encoder.pooler.dense",
        #"lm_head",
        "ln_1",
        "ln_2",
        #"transformer.wte",
        #"transformer.wpe",
    ])

    
def lora_attn(model: Model) -> None:
    target_modules = [
        "attention.query",
        "attention.key",
        "attention.value",
        "attn.c_attn",
        "attn.c_proj",
        "crossattention.c_attn",
        "crossattention.q_attn",
        "crossattention.c_proj",
    ]

    config = LoraConfig(
        r=32,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=target_modules,
        fan_in_fan_out=True,
    )
    
    model.load_peft_model(config)


def lora_all_linear(model: Model) -> None:
    target_modules = [
        "attention.query",
        "attention.key",
        "attention.value",
        "output.dense",
        "intermediate.dense",
        "pooler.dense",
        "attn.c_attn",
        "attn.c_proj",
        "crossattention.c_attn",
        "crossattention.q_attn",
        "crossattention.c_proj"
        "mlp.c_fc",
        "mlp.c_proj",
    ]

    config = LoraConfig(
        r=32,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=target_modules,
        fan_in_fan_out=True,
    )
    
    model.load_peft_model(config)


def wise(fine_tuned_model: Model, pre_trained_model: Model,
         alpha: float) -> None:
    theta_0 = pre_trained_model.model.state_dict()
    theta_1 = fine_tuned_model.model.state_dict()
    
    if set(theta_0.keys()) != set(theta_1.keys()):
        msg = "Incompatible models given to the WiSE function."
        raise ValueError(msg)

    theta = {
        key: (1-alpha) * theta_0[key] + alpha * theta_1[key]
        for key in theta_0
    }

    fine_tuned_model.model.load_state_dict(theta)
    fine_tuned_model.model.to("cuda:0") # type: ignore
    fine_tuned_model.__device = "cuda:0" # type: ignore # noqa: SLF001


def weight_word_embeddings() -> None:
    pass
