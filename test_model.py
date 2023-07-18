import pytest
import torch

from model import GPTConfig, GPT


@pytest.fixture
def model_config():
    return GPTConfig()


@pytest.fixture
def batch_ids():
    ids = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(1, -1).repeat(2, 1)
    return ids


def test_baseline(model_config, batch_ids):
    model = GPT(model_config)

    logits, loss = model(batch_ids)
    # the output will (batch x 1 x num_tokens) because the logits are only about the last position:
    # (see inference-time mini-optimization in model.py)
    assert logits.shape == (batch_ids.size(0), 1, model_config.vocab_size)

def test_alibi(model_config, batch_ids):
    model_config.use_alibi = True
    model = GPT(model_config)

    logits, loss = model(batch_ids)
    # the output will (batch x 1 x num_tokens) because the logits are only about the last position:
    # (see inference-time mini-optimization in model.py)
    assert logits.shape == (batch_ids.size(0), 1, model_config.vocab_size)
