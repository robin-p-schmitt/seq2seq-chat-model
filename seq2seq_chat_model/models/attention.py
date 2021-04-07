import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Attention(nn.Module):
    """Base class of different attention mechanisms.

    In order to parallelize multi-head attention, grouped convolutional
    layers are used to represent parallel Dense layers.

    Args:
        d_key_val (int): number of dimensions of the keys and values.
        d_query (int): number of dimension of the queries.
        d_k (int): number of dimensions the keys and values are mapped to.
        d_v (int): number of dimensions the queries are mapped to.
        n_heads (int): number of attention heads used.
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

            self.d_k = self.d_v = d_key_val // n_heads

        self.key_heads = nn.Conv1d(
            in_channels=d_key_val * n_heads,
            out_channels=self.d_k * n_heads,
            kernel_size=1,
            groups=n_heads,
        )
        self.query_heads = nn.Conv1d(
            in_channels=d_query * n_heads,
            out_channels=self.d_k * n_heads,
            kernel_size=1,
            groups=n_heads,
        )
        self.val_heads = nn.Conv1d(
            in_channels=d_key_val * n_heads,
            out_channels=self.d_v * n_heads,
            kernel_size=1,
            groups=n_heads,
        )

        self.projection = nn.Linear(n_heads * self.d_v, d_key_val)

    def project_heads(self, keys, vals, queries):
        """Returns the projected keys, values and queries."""

        # merge batch and seq dimension, repeat keys for n_heads-times and
        # apply the multiple projection heads
        keys_heads = self.key_heads(
            keys.view(-1, self.d_key_val, 1).repeat(1, self.n_heads, 1)
        )
        # re-view keys into shape (batch, sequence, n_heads, d_k)
        keys_heads = keys_heads.view(
            keys.size(0), keys.size(1), self.n_heads, self.d_k
        )

        # same as above but for values and queries
        vals_heads = self.val_heads(
            vals.view(-1, self.d_key_val, 1).repeat(1, self.n_heads, 1)
        )
        vals_heads = vals_heads.view(
            vals.size(0), vals.size(1), self.n_heads, self.d_v
        )
        queries_heads = self.query_heads(
            queries.view(-1, self.d_query, 1).repeat(1, self.n_heads, 1)
        )
        queries_heads = queries_heads.view(
            queries.size(0), queries.size(1), self.n_heads, self.d_k
        )

        return keys_heads, vals_heads, queries_heads


class AdditiveAttention(Attention):
    """Implementation of additive attention as described in
    https://arxiv.org/pdf/1409.0473.pdf.

    The scoring function is given by:
        score(query, key) = v^T * tanh(W * query + U * key)
    """

    def __init__(self, d_key_val, d_query, d_k=None, d_v=None, n_heads=1):
        super().__init__(d_key_val, d_query, d_k, d_v, n_heads)

        self.v = nn.Linear(self.d_k, 1)
        self.W = nn.Linear(self.d_k, self.d_k)
        self.U = nn.Linear(self.d_k, self.d_k)

    def forward(self, keys, vals, queries):
        # obtain different representations from attention heads
        keys_heads, vals_heads, queries_heads = self.project_heads(
            keys, vals, queries
        )

        # apply the scoring function to the keys and queries
        # scores is of shape (batch, query_seq, key_seq, n_heads, 1)
        # and assigns every query token a score for every key token
        scores = self.v(
            torch.tanh(
                self.W(queries_heads[:, :, None]) + self.U(keys_heads[:, None])
            )
        )
        # switch key_seq with n_heads
        scores = scores.permute(0, 1, 3, 2, 4)
        # apply softmax over key sequence and squeeze last dimension
        scores = F.softmax(torch.squeeze(scores, -1), -1)

        # permute the values to get the same order as the score
        # and multiply scores with the values to obtain weighted values
        context = (
            vals_heads.permute(0, 2, 1, 3)[:, None] * scores[:, :, :, :, None]
        )
        # sum over weighted values and concatenate context vectors of
        context = torch.sum(context, dim=-2)
        # concatenate attention heads
        context = context.view(*context.shape[:-2], -1)
        # project concatenated contexts to key/value dimension
        context = self.projection(context)

        return context


class DotProductAttention(Attention):
    """Implementation of (scaled) dot-product attention as described in
    https://arxiv.org/pdf/1706.03762.pdf and
    https://arxiv.org/pdf/1508.04025.pdf.

    The scoring function is given by:
        score(query, key) = query^T * key
    and is optionally scaled by d_k, which is the dimensionality of the
    queries and keys.
    """

    def __init__(
        self, d_key_val, d_query, d_k=None, d_v=None, n_heads=1, scaled=True
    ):
        super().__init__(d_key_val, d_query, d_k, d_v, n_heads)

        self.scaling_factor = 1

        if scaled:
            self.scaling_factor = self.d_k

    def forward(self, keys, vals, queries):
        # obtain different representations from attention heads
        keys_heads, vals_heads, queries_heads = self.project_heads(
            keys, vals, queries
        )

        # obtain three dimensional keys, queries and heads with shape
        # (batch * n_heads, seq_len, size)
        # and transpose the keys
        keys_heads = keys_heads.permute(0, 2, 3, 1)
        keys_heads = keys_heads.reshape(-1, *keys_heads.shape[2:])
        queries_heads = queries_heads.permute(0, 2, 1, 3)
        queries_heads = queries_heads.reshape(-1, *queries_heads.shape[2:])
        vals_heads = vals_heads.permute(0, 2, 1, 3)
        vals_heads = vals_heads.reshape(-1, *vals_heads.shape[2:])

        # obtain scores as batch matrix multiplication of queries and keys
        scores = torch.bmm(queries_heads, keys_heads) / self.scaling_factor
        scores = F.softmax(scores, dim=-1)

        # obtain context vectors as batch matmul of scores and values
        # context is of shape (batch * n_heads, n_queries, d_v)
        context = torch.bmm(scores, vals_heads)
        # separate batch and heads again and switch head dimension with
        # seq_len dimension
        context = context.view(
            -1, self.n_heads, queries_heads.shape[1], self.d_v
        ).permute(0, 2, 1, 3)
        # concatenate heads and project to key/value dimension
        context = context.reshape(*context.shape[:2], -1)
        context = self.projection(context)

        return context
