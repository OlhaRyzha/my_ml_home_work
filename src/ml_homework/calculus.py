from collections.abc import Callable


def forward_derivative(
    function: Callable[[float], float], x: float, step: float = 1e-5
) -> float:
    """Approximate a derivative with a first-order forward difference."""
    if step <= 0:
        raise ValueError("step must be positive")
    return (function(x + step) - function(x)) / step
