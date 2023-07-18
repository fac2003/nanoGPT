import torch
from torch import Tensor


def initialize_constant_bias_term(T: int):
    """
    Calculate the constant matrix for sequence length T. Here, we favor readability over efficiency,
    because the matrix is constant and only calculated once per sequence length (e.g., typically
    once per training run).
    """
    constant_terms = []
    for i in range(T):
        num_zeroes = i + 1
        zeroes = torch.zeros(num_zeroes)
        if num_zeroes == T:
            constant_terms.append(zeroes)
        else:
            constant_terms.append(torch.cat([zeroes, torch.arange(-1, -(T - num_zeroes + 1), -1)]))
    constant_term = torch.stack(constant_terms).transpose(0, 1)
    return constant_term


class AlibiApproach():
    def __init__(self, num_heads: int):
        self.num_heads = num_heads
        # to be defined on first call to get_bias_term
        self.sequence_length = None
        # one m value per attention head, we start at 2^(-8/num_heads) and increase following values
        # in geometric progression using the ratio (initial value):
        ratio = pow(2.0, (-8 / num_heads))
        self.m = [ratio]
        for i in range(num_heads - 1):
            self.m.append(ratio * self.m[-1])
        self.m = torch.tensor(self.m).reshape(1, num_heads, 1, 1)
        assert len(self.m.view(-1)) == num_heads, 'm must have one value per attention head'
        # to be defined on first call to get_bias_term, but cached per sequence length, which is useful
        # for testing the model with different sequence lengths
        self.constant_bias_terms = {}

    def get_bias_term(self, pre_softmax_attention: Tensor) -> Tensor:
        """
        Return the ALIBI bias term, a matrix of shape ( batch_size , num_heads, sequence_length, sequence_length), initialized
        per the paper.
        """
        batch_size, nh, T, T = pre_softmax_attention.shape
        assert nh == self.num_heads, 'number of heads must match with dimensions of argument to get_bias_term.'
        if T not in self.constant_bias_terms.keys():
            constant_term = initialize_constant_bias_term(T).to(pre_softmax_attention.device)

            self.constant_bias_terms[T] = constant_term

        return pre_softmax_attention + self.constant_bias_terms[T].repeat(batch_size, 1, 1, 1) * \
            self.m.repeat(batch_size, 1, 1, 1)
