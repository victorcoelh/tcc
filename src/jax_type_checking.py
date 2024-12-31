from jaxtyping import install_import_hook

with install_import_hook("src", "beartype.beartype"):
    from src.data_loading import dataset, preprocessing  # noqa: F401
    from src.fine_tuning import trainer  # noqa: F401
    from src.models import basemodel, vitgpt2  # noqa: F401
    from src.utils import type_hints  # noqa: F401


def main() -> None:
    print("Successful! All array types are matching.")  # noqa: T201


if __name__ == "__main__":
    main()
