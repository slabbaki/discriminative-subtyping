"""Utility functions for building contextual layers."""

from tensorflow.python.keras.engine.base_layer_utils import make_variable


def make_custom_getter(custom_variables):
    """Provides a custom getter for the given variables.

    Args:
        custom_variables: A dict of tensors replacing the trainable variables.

    Returns:
        The return a custom getter.
    """

    def custom_getter(name, **kwargs):
        if name in custom_variables:
            variable = custom_variables[name]
        else:
            variable = make_variable(name, **kwargs)
        return variable

    return custom_getter
