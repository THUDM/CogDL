from abc import ABCMeta, abstractmethod


class Attack(metaclass=ABCMeta):
    r"""

    Description
    -----------
    Abstract class for graph adversarial attack.

    """

    @abstractmethod
    def attack(self, model, adj, features, **kwargs):
        r"""

        Parameters
        ----------
        model : torch.nn.module
            Model implemented based on ``torch.nn.module``.
        adj : scipy.sparse.csr.csr_matrix
            Adjacency matrix in form of ``N * N`` sparse matrix.
        features : torch.FloatTensor
            Features in form of ``N * D`` torch float tensor.
        kwargs :
            Keyword-only arguments.

        """


class ModificationAttack(Attack):
    r"""

    Description
    -----------
    Abstract class for graph modification attack.

    """

    @abstractmethod
    def attack(self, **kwargs):
        """

        Parameters
        ----------
        kwargs :
            Keyword-only arguments.

        """

    @abstractmethod
    def modification(self, **kwargs):
        """

        Parameters
        ----------
        kwargs :
            Keyword-only arguments.

        """


class InjectionAttack(Attack):
    r"""

    Description
    -----------
    Abstract class for graph injection attack.

    """

    @abstractmethod
    def attack(self, **kwargs):
        """

        Parameters
        ----------
        kwargs :
            Keyword-only arguments.

        """

    @abstractmethod
    def injection(self, **kwargs):
        """

        Parameters
        ----------
        kwargs :
            Keyword-only arguments.

        """

    @abstractmethod
    def update_features(self, **kwargs):
        """

        Parameters
        ----------
        kwargs :
            Keyword-only arguments.

        """


class EarlyStop(object):
    r"""

    Description
    -----------
    Strategy to early stop attack process.

    """

    def __init__(self, patience=1000, epsilon=1e-4):
        r"""

        Parameters
        ----------
        patience : int, optional
            Number of epoch to wait if no further improvement. Default: ``1000``.
        epsilon : float, optional
            Tolerance range of improvement. Default: ``1e-4``.

        """
        self.patience = patience
        self.epsilon = epsilon
        self.min_score = None
        self.stop = False
        self.count = 0

    def __call__(self, score):
        r"""

        Parameters
        ----------
        score : float
            Value of attack acore.

        """
        if self.min_score is None:
            self.min_score = score
        elif self.min_score - score > 0:
            self.count = 0
            self.min_score = score
        elif self.min_score - score < self.epsilon:
            self.count += 1
            if self.count > self.patience:
                self.stop = True
