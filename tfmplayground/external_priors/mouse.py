from tabicl.prior._dataset import DisablePrinting, Prior
from torch import Tensor
from torch.utils.data import IterableDataset


class MousePrior(Prior):
    """Class for generating synthetic tabular datasets on-the-fly."""
    
    def __init__(self):
        pass


class MousePriorDataset(IterableDataset):
    """Main dataset class that provides an infinite iterator over synthetic tabular datasets.
    
    Parameters
    ----------
    batch_size : int, default=256
        Total number of datasets to generate per batch.

    batch_size_per_gp : int, default=4
        Number of datasets per group, sharing similar characteristics.

    batch_size_per_subgp : int, optional
        Number of datasets per subgroup, with more similar causal structures.
        If None, defaults to batch_size_per_gp.

    min_features : int, default=2
        Minimum number of features per dataset.

    max_features : int, default=100
        Maximum number of features per dataset.

    max_classes : int, default=10
        Maximum number of target classes.

    min_seq_len : int, optional
        Minimum samples per dataset. If None, uses max_seq_len directly.

    max_seq_len : int, default=1024
        Maximum samples per dataset.

    log_seq_len : bool, default=False
        If True, sample sequence length from a log-uniform distribution.

    seq_len_per_gp : bool, default=False
        If True, sample sequence length per group, allowing variable-sized datasets.

    min_train_size : int or float, default=0.1
        Position or ratio for train/test split start. If int, absolute position.
        If float between 0 and 1, specifies a fraction of sequence length.

    max_train_size : int or float, default=0.9
        Position or ratio for train/test split end. If int, absolute position.
        If float between 0 and 1, specifies a fraction of sequence length.

    replay_small : bool, default=False
        If True, occasionally sample smaller sequence lengths with
        specific distributions to ensure model robustness on smaller datasets.

    n_jobs : int, default=-1
        Number of parallel jobs to run (-1 means using all processors).

    num_threads_per_generate : int, default=1
        Number of threads per job for dataset generation.

    device : str, default="cpu"
        Computation device ('cpu' or 'cuda').
    """

    def __init__(
        self,
        batch_size: int = 256,
        batch_size_per_gp: int = 4,
        batch_size_per_subgp: int | None = None,
        min_features: int = 2,
        max_features: int = 100,
        max_classes: int = 10,
        min_seq_len: int | None = None,
        max_seq_len: int = 1024,
        log_seq_len: bool = False,
        seq_len_per_gp: bool = False,
        min_train_size: int | float = 0.1,
        max_train_size: int | float = 0.9,
        replay_small: bool = False,
        n_jobs: int = -1,
        num_threads_per_generate: int = 1,
        device: str = "cpu",
    ):
        super().__init__()
        self.prior = MousePrior()

        self.batch_size = batch_size
        self.batch_size_per_gp = batch_size_per_gp
        self.batch_size_per_subgp = batch_size_per_subgp or batch_size_per_gp
        self.min_features = min_features
        self.max_features = max_features
        self.max_classes = max_classes
        self.min_seq_len = min_seq_len
        self.max_seq_len = max_seq_len
        self.log_seq_len = log_seq_len
        self.seq_len_per_gp = seq_len_per_gp
        self.min_train_size = min_train_size
        self.max_train_size = max_train_size
        self.device = device

    def get_batch(self, batch_size: int | None = None) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Generate a new batch of datasets.

        Parameters
        ----------
        batch_size : int, optional
            If provided, overrides the default batch size for this call.

        Returns
        -------
        X : Tensor or NestedTensor
            1. For SCM-based priors:
             - If seq_len_per_gp=False, shape is ``(batch_size, seq_len, max_features)``.
             - If seq_len_per_gp=True, returns a NestedTensor.

            2. For DummyPrior, random Gaussian values of
            ``(batch_size, seq_len, max_features)``.

        y : Tensor or NestedTensor
            1. For SCM-based priors:
             - If seq_len_per_gp=False, shape is ``(batch_size, seq_len)``.
             - If seq_len_per_gp=True, returns a NestedTensor.

            2. For DummyPrior, random class labels of ``(batch_size, seq_len)``.

        d : Tensor
            Number of active features per dataset of shape ``(batch_size,)``.

        seq_lens : Tensor
            Sequence length for each dataset of shape ``(batch_size,)``.

        train_sizes : Tensor
            Position for train/test split for each dataset of shape ``(batch_size,)``.
        """
        return self.prior.get_batch(batch_size)

    def __iter__(self) -> "MousePriorDataset":
        """Return an iterator that yields batches indefinitely.

        Returns
        -------
        self
            Returns self as an iterator.
        """
        return self

    def __next__(self) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Return the next batch from the iterator.

        Since this is an infinite iterator, it never raises StopIteration
        and instead continuously generates new synthetic data batches.
        """
        with DisablePrinting():
            return self.get_batch()

    def __repr__(self) -> str:
        """Return a string representation of the dataset.

        Provides a detailed view of the dataset configuration for debugging
        and logging purposes.

        Returns
        -------
        str
            A formatted string with dataset parameters.
        """
        return (
            f"PriorDataset(\n"
            f"  prior_type: {self.prior_type}\n"
            f"  batch_size: {self.batch_size}\n"
            f"  batch_size_per_gp: {self.batch_size_per_gp}\n"
            f"  features: {self.min_features} - {self.max_features}\n"
            f"  max classes: {self.max_classes}\n"
            f"  seq_len: {self.min_seq_len or 'None'} - {self.max_seq_len}\n"
            f"  sequence length varies across groups: {self.seq_len_per_gp}\n"
            f"  train_size: {self.min_train_size} - {self.max_train_size}\n"
            f"  device: {self.device}\n"
            f")"
        )
