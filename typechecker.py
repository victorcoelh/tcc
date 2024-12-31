from beartype.claw import beartype_package
from jaxtyping import install_import_hook

# decorate `@jaxtyped(typechecker=typeguard.typechecked)`
with install_import_hook("src", "typeguard.typechecked"):
    from src import dataset
    beartype_package("src")
