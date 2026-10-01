"""Punto de entrada del proyecto: python main.py"""
import random

import numpy as np

SEED = 42


def set_seed(seed: int = SEED) -> None:
    """Fija las semillas para que los resultados sean reproducibles."""
    random.seed(seed)
    np.random.seed(seed)


def main() -> None:
    set_seed()
    print("Pipeline aún no implementado.")


if __name__ == "__main__":
    main()
