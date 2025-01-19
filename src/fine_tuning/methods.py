from src.models.basemodel import Model


def clip_fit() -> None:
    pass


def clip_adapter() -> None:
    pass


def wise(fine_tuned_model: Model, pre_trained_model: Model,
         weight_a: float, weight_b: float) -> None:
    pre_trained_params = list(pre_trained_model.model.parameters())

    for i, parameter in enumerate(fine_tuned_model.model.parameters()):
        param_sum = ((parameter.data * weight_a) + (pre_trained_params[i] * weight_b))
        parameter.data = param_sum / (weight_a + weight_b)


def weight_word_embeddings() -> None:
    pass
