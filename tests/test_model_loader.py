from types import SimpleNamespace

import torch

from src.model_loader import _features


def test_tensor_output_passes_through():
    t = torch.ones(1, 4)
    assert _features(t) is t


def test_model_output_uses_pooler_output():
    # Regression: transformers 5.x wraps features in BaseModelOutputWithPooling
    t = torch.ones(1, 4)
    assert _features(SimpleNamespace(pooler_output=t, last_hidden_state=None)) is t
