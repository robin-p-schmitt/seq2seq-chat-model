import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC
from abc import abstractmethod

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Attention(nn.Module, ABC):
    """Base class of different attention mechanisms.

    In order to parallelize multi-head attention, grouped convolutional
    layers are used to represent parallel Dense layers.

    :param d_key_val: dimensionality of the keys and values.
    :type d_key_val: int
    :param d_query: dimensionality of the queries.
    :type d_query: int
    :param d_k: dimensionality of the key/query attention subspace.
        If `None`, :param:`d_k` defaults to :param:`d_key_val`/`8`.
    :type d_k: int, optional
    :param d_v: dimensionality of the value attention subspace.
        If `None`, :param:`d_v` defaults to :param:`d_key_val`/8.
    :type d_v: int, optional
    :param n_heads: number of attention heads used. Defaults to 1.
    :type n_heads: int, optional
    """

    def __init__(self, d_key_val, d_query, d_k=None, d_v=None, n_heads=1):
        super(Attention, self).__init__()

        self.d_key_val = d_key_val
        self.d_query = d_query
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads

        if d_k is None or d_v is None:
            if d_key_val % n_heads != 0:
                raise ValueError("d_key_val is not divisible by n_heads")

            if d_k is None:
                self.d_k = d_key_val // n_heads
            if d_v is None:
                self.d_v = d_key_val // n_heads

        self.key_projections = nn.Conv1d(
            in_channels=d_key_val * n_heads,
            out_channels=self.d_k * n_heads,
            kernel_size=1,
            groups=n_heads,
        )
        self.query_projections = nn.Conv1d(
            in_channels=d_query * n_heads,
            out_channels=self.d_k * n_heads,
            kernel_size=1,
            groups=n_heads,
        )
        self.val_projections = nn.Conv1d(
            in_channels=d_key_val * n_heads,
            out_channels=self.d_v * n_heads,
            kernel_size=1,
            groups=n_heads,
        )

        self.projection = nn.Linear(n_heads * self.d_v, d_key_val)

    def _get_attention_heads(self, tensor, projection):
        """Returns the projected keys, values and queries."""

        # merge batches and sequences and add length of
        # signal sequence dimension
        heads = tensor.view(-1, tensor.shape[-1], 1)
        # repeat the vectors in the 2nd dimension for
        # n_heads times
        heads = heads.repeat(1, self.n_heads, 1)
        # project the different repitions using different weights
        heads = projection(heads)
        # separate batches and sequences and
        # the different projected heads
        heads = heads.view(*tensor.shape[:2], self.n_heads, -1)
        # switch head and sequence dimension
        heads = heads.transpose(1, 2)
        # merge batch and head dimension
        heads = heads.reshape(-1, *heads.shape[2:])

        return heads

    @abstractmethod
    def _scoring_function(self, query_heads, key_heads):
        ...

    def forward(self, keys, vals, queries):
        # obtain different representations
        # of shape (batch * n_heads, seq_len, size)
        key_heads = self._get_attention_heads(keys, self.key_projections)
        val_heads = self._get_attention_heads(vals, self.val_projections)
        query_heads = self._get_attention_heads(
            queries, self.query_projections
        )

        scores = self._scoring_function(query_heads, key_heads)

        # apply softmax over key sequence and squeeze last dimension
        scores = F.softmax(torch.squeeze(scores, -1), -1)

        context = torch.bmm(scores, val_heads)

        context = context.view(-1, self.n_heads, *context.shape[1:])
        context = context.transpose(1, 2)
        # concatenate attention heads
        context = context.reshape(*context.shape[:2], -1)
        # project concatenated contexts to key/value dimension
        context = self.projection(context)

        return context


class AdditiveAttention(Attention):
    """Implementation of additive attention as described in
    https://arxiv.org/pdf/1409.0473.pdf.

    The scoring function is given by:

        score(key, query) = v^T * tanh(W * query + U * key)
    """

    def __init__(self, d_key_val, d_query, d_k=None, d_v=None, n_heads=1):
        super().__init__(d_key_val, d_query, d_k, d_v, n_heads)

        self.v = nn.Linear(self.d_k, 1)
        self.W = nn.Linear(self.d_k, self.d_k)
        self.U = nn.Linear(self.d_k, self.d_k)

    def _scoring_function(self, query_heads, key_heads):
        scores = self.v(
            torch.tanh(
                self.W(query_heads[:, :, None]) + self.U(key_heads[:, None])
            )
        )
        return scores


class GeneralAttention(Attention):
    """Implementation of general attention as described in
    https://arxiv.org/pdf/1508.04025.pdf.

    The scoring function is given by:

        score(key, query) = query^T * W * key
    """

    def __init__(self, d_key_val, d_query, d_k=None, d_v=None, n_heads=1):
        super().__init__(d_key_val, d_query, d_k, d_v, n_heads)

        self.W = nn.Linear(self.d_k, self.d_k)

    def _scoring_function(self, query_heads, key_heads):
        scores = torch.bmm(query_heads, self.W(key_heads).transpose(1, 2))
        return scores


class DotProductAttention(Attention):
    """Implementation of dot-product attention as described in
    https://arxiv.org/pdf/1508.04025.pdf.

    The scoring function is given by:

        score(query, key) = query^T * key

    """

    def _scoring_function(self, query_heads, key_heads):
        scores = torch.bmm(query_heads, torch.transpose(key_heads, 1, 2))
        return scores


class ScaledDotProductAttention(Attention):
    """Implementation of scaled dot-product attention
    as described in
    https://arxiv.org/pdf/1706.03762.pdf.

    The scoring function is given by:

        score(query, key) = (query^T * key) / d_k

    """

    def _scoring_function(self, query_heads, key_heads):
        scores = torch.bmm(query_heads, torch.transpose(key_heads, 1, 2))
        return scores


class CosineAttention(Attention):
    """Implementation of cosine attention as described in
    https://arxiv.org/pdf/1410.5401.pdf.

    The scoring function is given by:

        score(query, key) = (query^T * key) / (||query|| * ||key||)

    """

    def _scoring_function(self, query_heads, key_heads):
        queries_norm = query_heads / query_heads.norm(dim=-1, keepdim=True)
        keys_norm = key_heads / key_heads.norm(dim=-1, keepdim=True)
        scores = torch.bmm(queries_norm, torch.transpose(keys_norm, 1, 2))
        return scores
