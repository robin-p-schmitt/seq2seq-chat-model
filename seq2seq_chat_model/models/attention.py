"""This module contains implementations of different attention mechanisms.

The base class of all attention classes is the abstract ``Attention`` base
class.

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC
from abc import abstractmethod
from seq2seq_chat_model.models.utils import attention_mask

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
attention_types = ["additive"]


class Attention(nn.Module, ABC):
    """Base class of all other attention classes.

    All implemented attention classes need to extend this class. Here, all
    basic attention parameters are initialized and the attention mechanism
    is defined in the ``forward`` function.

    For multi head attention, grouped convolutional layers are used to
    enable parallelization across attention heads. For this, the input
    vector is repeated for ``n_heads`` times. Then, each "repetition" is
    processed by a 1D convolutional layer with kernel size 1 and
    dim(output) channels. This corresponds to ``n_heads`` Dense layers,
    each with dim(input) input neurons and dim(output) output neurons.

    Attributes: 
        d_key_val (int): the dimensionality of the keys and values.
        d_query (int): the dimensionality of the queries. 
        d_k (int, optional): the dimensionality of the projected keys and queries.
            Defaults to ``d_key_val`` / ``n_heads``.
        d_v (int, optional): the dimensionality of the projected queries.
            Defaults to ``d_key_val`` / ``n_heads``.
        n_heads (int, optional): the number of attention heads used. Defaults to 1.
    """

    def __init__(
        self,
        d_key_val,
        d_query,
        d_k=None,
        d_v=None,
        n_heads=1,
        masked=False,
    ):
        """Initialize all basic variables needed for any attention mechanism.

        Raises:
            ValueError: If ``d_k`` or ``d_v`` is ``None`` and ``d_key_val``
            is not divisible by ``n_heads``.
        """
        super(Attention, self).__init__()

        self.d_key_val = d_key_val
        self.d_query = d_query
        self.d_k = d_k
        self.d_v = d_v
        self.n_heads = n_heads
        self.masked = masked

        # initialize ``d_k`` and ``d_v`` as ``d_key_val`` / ``n_heads`` if
        # not given
        if d_k is None or d_v is None:
            if d_key_val % n_heads != 0:
                raise ValueError("d_key_val is not divisible by n_heads")

            if d_k is None:
                self.d_k = d_key_val // n_heads
            if d_v is None:
                self.d_v = d_key_val // n_heads

        # convolutional projection layers which each represent one
        # attention head

        # for keys
        self.key_projections = nn.Conv1d(
            in_channels=d_key_val * n_heads,
            out_channels=self.d_k * n_heads,
            kernel_size=1,
            groups=n_heads,
        )
        # for queries
        self.query_projections = nn.Conv1d(
            in_channels=d_query * n_heads,
            out_channels=self.d_k * n_heads,
            kernel_size=1,
            groups=n_heads,
        )
        # for values
        self.val_projections = nn.Conv1d(
            in_channels=d_key_val * n_heads,
            out_channels=self.d_v * n_heads,
            kernel_size=1,
            groups=n_heads,
        )

        # last projection layer which maps the concatenation of attention
        # heads to the output of the attention mechanism
        self.projection = nn.Linear(n_heads * self.d_v, d_key_val)

        self.attention_scores = None

    def get_attention_scores(self):
        return self.attention_scores

    def _get_attention_heads(self, tensor, projection):
        """Apply attention heads to the input tensor.

        Args:
            tensor (torch.tensor):
                Input tensor of shape (batch, seq_len, size)
            projection (function):
                Function to use for projection.

        Returns:
            torch.tensor:
            Projected version of the input tensor of shape
            (batch * n_heads, seq_len, size_projected)
        """

        # merge batches and sequences and add length of
        # signal sequence dimension
        heads = tensor.view(-1, tensor.shape[-1], 1)
        # repeat the vectors in the 2nd dimension for
        # n_heads times
        heads = heads.repeat(1, self.n_heads, 1)
        # project the different repitions using different weights
        heads = projection(heads)
        # separate batches and sequences and
        heads = heads.view(*tensor.shape[:2], self.n_heads, -1)
        # switch head and sequence dimension
        heads = heads.transpose(1, 2)
        # merge batch and head dimension
        heads = heads.reshape(-1, *heads.shape[2:])

        return heads

    @abstractmethod
    def _scoring_function(self, query_heads, key_heads):
        """Defines the attention scoring function.

        Args:
            query_heads (torch.tensor): queries of shape
                (batch * n_heads, query_seq_len, d_k)

            key_heads (torch.tensor): keys of shape
                (batch * n_heads, key_seq_len, d_k)

        Returns:
            torch.tensor: unnormalized scores of shape
                (batch * n_heads, query_seq_len, key_seq_len, 1)
                which assigns every token in the query sequence
                a score for every token in the key sequence.
        """
        ...

    def forward(self, keys, vals, queries):
        """Define general attention mechanism.

        First, the projections of the attention heads are obtained for
        keys, queries and values. Then, the attention scores are obtained.
        After that, the weighted sum of values is obtained. Lastly, the
        different heads are concatenated and projected to the original
        dimensionality of keys and values.

        Args:
            keys (torch.tensor): sequence of keys of shape (batch,
                key_seq_len, d_key_val)
            vals (torch.tensor): sequence of values of shape (batch,
                val_seq_len, d_key_val)
            queries (torch.tensor): sequence of queries of shape (batch,
                query_seq_len, d_query)

        Returns:
            torch.tensor: weighted context vector of shape
                (batch, query_seq_len, d_key_val)
        """
        # obtain projections
        key_heads = self._get_attention_heads(keys, self.key_projections)
        val_heads = self._get_attention_heads(vals, self.val_projections)
        query_heads = self._get_attention_heads(
            queries, self.query_projections
        )
        
        # obtain unnormalized scores
        scores = self._scoring_function(query_heads, key_heads)

        # if masked attention is used, mask out illegal connections before softmax
        if self.masked:
            if key_heads.shape[1] != query_heads.shape[1]:
                raise ValueError(
                    "When using masked attention, the length of keys and queries needs to be identical!"
                )
            scores = scores + attention_mask(query_heads.shape[1])[None]

        # apply softmax over key sequence and squeeze last dimension
        scores = F.softmax(scores, -1)
        self.attention_scores = scores

        # obtain weighted sum of values
        context = torch.bmm(scores, val_heads)

        # separate batch and head dimension and transpose heads and seq_len
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

    Attributes:
        v (torch.tensor): project vector to unnormalized score.
        W (torch.tensor): transform queries.
        U (torch.tensor): transform keys.
    """

    def __init__(
        self,
        d_key_val,
        d_query,
        d_k=None,
        d_v=None,
        n_heads=1,
        masked=False,
    ):
        super().__init__(
            d_key_val,
            d_query,
            d_k=d_k,
            d_v=d_v,
            n_heads=n_heads,
            masked=masked,
        )

        self.v = nn.Linear(self.d_k, 1)
        self.W = nn.Linear(self.d_k, self.d_k)
        self.U = nn.Linear(self.d_k, self.d_k)

    def _scoring_function(self, query_heads, key_heads):
        scores = self.v(
            torch.tanh(
                self.W(query_heads[:, :, None]) + self.U(key_heads[:, None])
            )
        )

        scores = scores.squeeze(-1)
        return scores


class GeneralAttention(Attention):
    """Implementation of general attention as described in
    https://arxiv.org/pdf/1508.04025.pdf.

    The scoring function is given by:

        score(key, query) = query^T * W * key

    Attributes:
        W (torch.tensor): transforms queries and keys.
    """

    def __init__(
        self,
        d_key_val,
        d_query,
        d_k=None,
        d_v=None,
        n_heads=1,
        masked=False,
    ):
        super().__init__(
            d_key_val,
            d_query,
            d_k=d_k,
            d_v=d_v,
            n_heads=n_heads,
            masked=masked,
        )

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
