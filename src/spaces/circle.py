"""Implementation of a space that represents circles in euclidean space."""
from typing import List, Optional, Sequence, SupportsFloat, Tuple, Type, Union

import numpy as np

import gymnasium.error
from gymnasium import logger
from gymnasium.spaces.space import Space


def _short_repr(arr: np.ndarray) -> str:
    """Create a shortened string representation of a numpy array.

    If arr is a multiple of the all-ones vector, return a string representation of the multiplier.
    Otherwise, return a string representation of the entire array.

    Args:
        arr: The array to represent

    Returns:
        A short representation of the array
    """
    if arr.size != 0 and np.min(arr) == np.max(arr):
        return str(np.min(arr))
    return str(arr)


def is_float_integer(var) -> bool:
    """Checks if a variable is an integer or float."""
    return np.issubdtype(type(var), np.integer) or np.issubdtype(type(var), np.floating)


class Circle(Space[np.ndarray]):
    r"""A bounded circle in :math:`\mathbb{R}^2`.


    """

    def __init__(
        self,
        radius: SupportsFloat,
        dtype: Type = np.float32,
        seed: Optional[Union[int, np.random.Generator]] = None,
    ):
        r"""Constructor of :class:`Circle`.

        Args:
            radius (SupportsFloat): the radius of the space.
            dtype: The dtype of the elements of the space. If this is an integer type, the :class:`Circle` is essentially a discrete space.
            seed: Optionally, you can use this argument to seed the RNG that is used to sample from the space.

        Raises:
            ValueError: If no radius is provided (shape is None then a value error is raised.
        """
        assert (
            dtype is not None
        ), "Circle dtype must be explicitly provided, cannot be None."
        self.dtype = np.dtype(dtype)

        assert ( #todo typecheck here
            radius is not None
        ), "Circle radius must be explicitly provided, cannot be None."
        self.radius = radius
        self._radius_sq = radius*radius
        self.bounded_below = True
        self.bounded_above = True

        self._shape: Tuple[int, ...] = (2,)
        # self.shape = (2,)
        if get_precision(type(radius)) > get_precision(self.dtype):  # type: ignore
            logger.warn(f"Box bound precision lowered by casting to {self.dtype}")
        # self.low = low.astype(self.dtype)
        # self.high = high.astype(self.dtype)

        # self.low_repr = _short_repr(self.low)
        # self.high_repr = _short_repr(self.high)

        super().__init__(self.shape, self.dtype, seed)

    @property
    def shape(self) -> Tuple[int, ...]:
        """Has stricter type than gym.Space - never None."""
        return self._shape

    @property
    def is_np_flattenable(self):
        """Checks whether this space can be flattened to a :class:`spaces.Circle`."""
        return True

    def is_bounded(self, manner: str = "both") -> bool:
        """Checks whether the box is bounded in some sense.

        Args:
            manner (str): One of ``"both"``, ``"below"``, ``"above"``.

        Returns:
            If the space is bounded

        Raises:
            ValueError: If `manner` is neither ``"both"`` nor ``"below"`` or ``"above"``
        """
        below = bool(np.all(self.bounded_below))
        above = bool(np.all(self.bounded_above))
        if manner == "both":
            return below and above
        elif manner == "below":
            return below
        elif manner == "above":
            return above
        else:
            raise ValueError(
                f"manner is not in {{'below', 'above', 'both'}}, actual value: {manner}"
            )

    def sample(self, mask: None = None) -> np.ndarray:
        r"""Generates a single random sample from the edge of the circle.

        In creating a sample of the circle, each coordinate is sampled (independently) from a uniform(-1,1)
        distribution. The result is then scaled by the circle radius.

        Args:
            mask: A mask for sampling values from the Circle space, currently unsupported.

        Returns:
            A sampled value from the Circle
        """
        if mask is not None:
            raise gym.error.Error(
                f"Circle.sample() cannot be provided a mask, actual value: {mask}"
            )

        sample = np.empty(self.shape)

        # Masking arrays which classify the coordinates according to interval
        # type
        bounded = self.bounded_below & self.bounded_above

        # Vectorized sampling

        sample[bounded] = self.np_random.uniform(
            low=-1, high=1, size=(2,)
        )*self.radius
        
        if self.dtype.kind == "i": #ripped directly from the Box equivalent
            sample = np.floor(sample) #almost certainly doesnt work properly

        return sample.astype(self.dtype)

    def contains(self, x) -> bool:
        """Return boolean specifying if x is a valid member of this space."""
        if not isinstance(x, np.ndarray):
            logger.warn("Casting input x to numpy array.")
            try:
                x = np.asarray(x, dtype=self.dtype)
            except (ValueError, TypeError):
                return False

        return bool(
            np.can_cast(x.dtype, self.dtype)
            and x.shape == self.shape
            and np.dot(x,x)-self._radius_sq == 0
        )

    def to_jsonable(self, sample_n):
        """Convert a batch of samples from this space to a JSONable data type."""
        return np.array(sample_n).tolist()

    def from_jsonable(self, sample_n: Sequence[Union[float, int]]) -> List[np.ndarray]:
        """Convert a JSONable data type to a batch of samples from this space."""
        return [np.asarray(sample) for sample in sample_n]

    def __repr__(self) -> str:
        """A string representation of this space.

        The representation will include bounds, shape and dtype.
        If a bound is uniform, only the corresponding scalar will be given to avoid redundant and ugly strings.

        Returns:
            A representation of the space
        """
        return f"Circle({self.radius}, {self.dtype})"

    def __eq__(self, other) -> bool:
        """Check whether `other` is equivalent to this instance. Doesn't check dtype equivalence."""
        return (
            isinstance(other, Circle)
            and (self.shape == other.shape)
            # and (self.dtype == other.dtype)
            and self.radius-other.radius == 0
        )

def get_precision(dtype) -> SupportsFloat:
    """Get precision of a data type."""
    if np.issubdtype(dtype, np.floating):
        return np.finfo(dtype).precision
    else:
        return np.inf


# todo: tests