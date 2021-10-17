"""Additional layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange

from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import (
    Embedding, GlobalMaxPooling1D, InputSpec, Layer, Lambda)


__all__ = [
    "OneHot",
    "GlobalMaxPooling1DMasked",
    "GlobalMaxPool1DMasked",
    "PadVector",
    "NormalizedEmbedding",
]


class OneHot(Lambda):
    """One-hot encoding layer (stateless)."""

    def __init__(self, output_dim, **kwargs):
        output_shape = lambda input_shape: list(input_shape) + [output_dim]
        super(OneHot, self).__init__(
            lambda x: K.one_hot(x, output_dim),
            output_shape=output_shape,
            **kwargs)


class GlobalMaxPooling1DMasked(GlobalMaxPooling1D):
    """Global max pooling operation for temporal data that support masks.

    # Input shape
        3D tensor with shape: `(samples, steps, features)`.

    # Output shape
        2D tensor with shape: `(samples, features)`.
    """

    def __init__(self, **kwargs):
        super(GlobalMaxPooling1DMasked, self).__init__(**kwargs)
        self.supports_masking = True

    def compute_mask(self, input, input_mask=None):
        """Make sure the the input mask doesn't propagate forward."""
        return None

    def call(self, x, mask=None):
        if mask is not None:
            mask_target_shape = []
            for dim in xrange(K.ndim(mask)):
                mask_target_shape.append(K.shape(mask)[dim])
            mask_target_shape = K.stack(mask_target_shape + [1])
            mask_reshaped = K.reshape(mask, mask_target_shape)
            x_masked = x * K.cast(mask_reshaped, K.floatx())
            return K.max(x_masked, axis=1)
        else:
            return super(GlobalMaxPooling1DMasked, self).call(x)


class NormalizedEmbedding(Embedding):
    """Embedding normalized by vocabulary frequencies."""

    def __init__(self, input_dim, output_dim, vocab_freqs=None, **kwargs):
        super(NormalizedEmbedding, self).__init__(
            input_dim, output_dim, **kwargs)

        self.vocab_freqs = vocab_freqs
        if self.vocab_freqs is not None:
            self.vocab_freqs = K.constant(
                vocab_freqs, dtype='float32', shape=(vocab_freqs.shape[0], 1))

    def _normalize(self, emb):
        weights = self.vocab_freqs / K.sum(self.vocab_freqs)
        mean = K.sum(weights * emb, 0, keepdims=True)
        var = K.sum(weights * K.pow(emb - mean, 2.), 0, keepdims=True)
        stddev = K.sqrt(1e-6 + var)
        return (emb - mean) / stddev

    def call(self, inputs):
        out = super(NormalizedEmbedding, self).call(inputs)
        if self.vocab_freqs is not None:
            out = self._normalize(out)
        return out


class PadVector(Layer):
    """Pads the input with n - 1 constant vectors.

    # Arguments
        n: integer, padding factor.

    # Input shape
        2D tensor of shape `(nb_samples, features)`.

    # Output shape
        3D tensor of shape `(nb_samples, n, features)`.
    """

    def __init__(self, n, constant_value=0.0, **kwargs):
        self.n = n
        self.constant_value = constant_value
        self.input_spec = [InputSpec(ndim=2)]
        super(PadVector, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.n, input_shape[1])

    def call(self, x, mask=None):
        x_shape = K.int_shape(x)
        x = K.reshape(x, shape=(-1, 1, x_shape[1]))
        return K.pad(x, [[0, 0], [0, self.n - 1], [0, 0]])

    def get_config(self):
        config = {'n': self.n}
        base_config = super(PadVector, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


# Aliases
GlobalMaxPool1DMasked = GlobalMaxPooling1DMasked
