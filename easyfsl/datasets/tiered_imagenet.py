from pathlib import Path

from .easy_set import EasySet

TIERED_IMAGENET_SPECS_DIR = Path("data/tiered_imagenet")


class TieredImageNet(EasySet):
    def __init__(self, split: str, **kwargs):
        """
        Build the tieredImageNet dataset for the specific split.
        Args:
            split: one of the available split (typically train, val, test).
        Raises:
            ValueError: if the specified split cannot be associated with a JSON spec file
                from tieredImageNet's specs directory
        """
        specs_file = TIERED_IMAGENET_SPECS_DIR / f"{split}.json"
        if not specs_file.is_file():
            raise ValueError(
                f"Could not find specs file {specs_file.name} in {TIERED_IMAGENET_SPECS_DIR}"
            )
        super().__init__(specs_file=specs_file, **kwargs)
