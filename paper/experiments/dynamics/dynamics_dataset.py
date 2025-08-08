from __future__ import annotations

import numpy as np
import torch

ArrayLike = np.ndarray | torch.Tensor
import logging

log = logging.getLogger(__name__)


class TrajectoryDataset(torch.utils.data.Dataset):
    """Class for a collection of context windows with tensor features."""

    def __init__(
        self,
        trajectories: list[ArrayLike] | ArrayLike,
        past_frames: int = 1,
        future_frames: int = 1,
        time_lag: int = 1,
        shuffle: bool = False,
        seed: int = 1234,
        **torch_kwargs,
    ):
        """Initialize a Dataset instance that can be passed to a torch.data.DataLoader.

        Args:
            trajectories: list([time, feat_dims*]) trajectories of potentially different lengths but same feature dimension.
            past_frames: (int) Number of past time-frames to return in each context window.
            future_frames: (int) Number of future time-frames to return in each context window.
            time_lag: (int) Time lag between successive context windows. Default to 1.
            shuffle: (bool) If True, shuffles the context windows. Default to False.
            seed: (int) Seed for the random number generator. Default to 1234.
            **torch_kwargs: (dict) Keyword arguments to pass to the backend.
                If backend='torch', for instance it is possible to specify the device and type of the data samples.
                If backend='numpy', it is possible to specify the dtype of the data samples
        """
        assert past_frames > 0, f"past_frames must be > 0, got {past_frames}"
        assert future_frames > 0, f"future_frames must be > 0, got {future_frames}"
        context_length = past_frames + future_frames

        if time_lag < 1:
            raise ValueError(f"time_lag must be >= 1, got {time_lag}")

        self._context_length = context_length
        self._past_frames = past_frames
        self._future_frames = future_frames
        self._time_lag = time_lag
        self._indices = []
        self._raw_data = []  # Variable containing the trajectories in the desired backed.

        # Convert trajectories to the desired backend. We copy data only once, and keep the original memory footprint.
        self._raw_data = trajectories  # [torch.tensor(traj, **torch_kwargs) for traj in trajectories]

        # Compute the list of indices (traj_idx, slice(start, end)) for each ContextWindow.
        for traj_idx, traj_data in enumerate(self._raw_data):
            context_window_slices = _slices_from_traj_len(
                time_horizon=len(traj_data), context_length=context_length, time_lag=time_lag
            )
            # Store a tuple of (traj_idx, context window slice) for each context window.
            self._indices.extend([(traj_idx, s) for s in context_window_slices])

        self._memory_footprint = None
        self._shuffled = False

        if shuffle:
            self.shuffle(seed=seed)

        if len(self) == 0:
            raise RuntimeError(f"No context windows of length {self.context_length} found on trajectory data")

    def shuffle(self, seed: int = None):
        """Shuffles the context windows."""
        if seed is not None:
            np.random.seed(seed)
        np.random.shuffle(self._indices)
        self._shuffled = True

    @property
    def context_length(self):
        return int(self._context_length)

    @property
    def time_lag(self):
        return int(self._time_lag)

    @property
    def is_shuffled(self):
        return self._shuffled

    @property
    def memory_footprint(self):
        """Returns the memory footprint of the dataset in bytes."""
        if self._memory_footprint is None:
            self._memory_footprint = sum(traj.element_size() * traj.nelement() for traj in self._raw_data)
        return self._memory_footprint

    def __len__(self):
        return len(self._indices)

    def __getitem__(self, idx):
        traj_idx, slice_idx = self._indices[idx]
        context_window = self._raw_data[traj_idx][slice_idx.start : slice_idx.stop]
        return context_window

    def __repr__(self):
        device = "cpu"
        if hasattr(self, "_raw_data "):
            if len(self._raw_data) > 0:
                device = self._raw_data[0].device
            return f"Memory use: {self.memory_footprint / 1e6:.2f} MB on {device}"
        else:
            return "TrajectoryContextDataset not initialized."

    def get_all_past_windows(self):
        """
        Returns a tensor of shape [n_samples, state_dim, past_frames] containing all past windows.
        Each window is a view into the original trajectory data (no memory duplication).
        """
        views = []
        for traj_idx, slice_idx in self._indices:
            s, e = slice_idx.start, slice_idx.stop
            past = self._raw_data[traj_idx][..., s : s + self._past_frames]
            views.append(past)
        return torch.stack(views)

    def get_all_future_windows(self):
        """
        Returns a tensor of shape [n_samples, state_dim, future_frames] containing all future windows.
        Each window is a view into the original trajectory data (no memory duplication).
        """
        views = []
        for traj_idx, slice_idx in self._indices:
            s, e = slice_idx.start, slice_idx.stop
            future = self._raw_data[traj_idx][..., s + self._past_frames : e]
            views.append(future)
        return torch.stack(views)


def _slices_from_traj_len(time_horizon: int, context_length: int, time_lag: int) -> list[slice]:
    """Returns the list of slices (start_time_idx, end_time_idx) for each context window in the trajectory.
    Args:
        time_horizon: (int) Number time-frames of the trajectory.
        context_length: (int) Number of time-frames per context window
        time_lag: (int) Time lag between successive context windows.
    Returns:
        list[slice]: List of slices for each context window.

    Examples
    --------
    >>> time_horizon, context_length, time_lag = 10, 4, 2
    >>> slices = _slices_from_traj_len(time_horizon, context_length, time_lag)
    >>> for s in slices:
    ...     print(f"start: {s.start}, end: {s.stop}")
    start: 0, end: 4
    start: 2, end: 6
    start: 4, end: 8
    start: 6, end: 10
    """
    slices = []
    for start in range(0, time_horizon - context_length + 1, time_lag):
        end = start + context_length
        slices.append(slice(start, end))

    return slices
