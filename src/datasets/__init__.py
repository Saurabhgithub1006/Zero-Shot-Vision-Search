from src.datasets.a2d2 import A2D2Dataset
from src.datasets.unsplash import UnsplashDataset

_REGISTRY = {
    "a2d2": A2D2Dataset,
    "unsplash": UnsplashDataset,
}


def get_dataset(name):
    if name not in _REGISTRY:
        raise ValueError(f"Unknown dataset '{name}'. Available: {', '.join(_REGISTRY)}")
    return _REGISTRY[name]()
