from math import sqrt

import pytest
import torch

from alibi import initialize_constant_bias_term, AlibiApproach
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


@pytest.fixture
def attention_matrix():
    batch_size = 1
    nh = 3
    T = 5
    return torch.randn(batch_size, nh, T, T)

def test_constant_bias_term(attention_matrix):
    bs, nh, T, T = attention_matrix.shape
    alibi_constrant_bias_term = initialize_constant_bias_term(T)
    assert alibi_constrant_bias_term.shape == (T, T)

def test_alibi_m_values():
    alibi = AlibiApproach(16)
    assert pytest.approx(alibi.m[:, 0, :, :].item()) == 1.0 / sqrt(2.0)
    assert pytest.approx(alibi.m[:, 1, :, :].item()) == pow(1.0 / sqrt(2.0), 2)

    alibi = AlibiApproach(8)
    assert pytest.approx(alibi.m[:, 0, :, :].item()) == 1.0 / 2
    assert pytest.approx(alibi.m[:, 1, :, :].item()) == 1 / pow(2, 2)
