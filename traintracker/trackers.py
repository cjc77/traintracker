from abc import ABC, abstractmethod
import random

from traintracker.util.defs import *
from traintracker.client import Client


def unique_id() -> Iterator[int]:
    """ Generate a continuous stream of unique IDs.

    First ID is random, next IDs are incremented from initial seed.
    
    Returns:
        int: a unique id, based on a randomly seeded initial ID.
    """
    seed_id = random.getrandbits(32)

    while True:
        yield seed_id
        seed_id += 1


class Tracker(ABC):
    """ Base class for all trackers. 

    Trackers are utilities that track various metrics
    regarding the performance of a model.
    """
    id_generator: Iterator[int] = unique_id()
    def __init__(self, plot_type: PlotType, name: str, client: Optional[Client] = None):
        """
        Args:
            plot_type (PlotType): type of plot that will be made by server if
                tracker is connected to a client
            name (str): the name of this tracker, e.g. "model 1 loss"
            client (Client or None): the client that the tracker is connected to
        """
        self._plot_type: PlotType = plot_type
        self._client: Optional[Client] = client
        self._name: str = name
        self._id: int = next(self.id_generator)

    @property
    def plot_type(self) -> PlotType:
        return self._plot_type

    @property
    def id(self) -> int:
        return self._id

    def connect_client(self, client: Client) -> None:
        """ Connect this tracker to a client.
        
        Args:
            client (Client): the client to which this tracker will send its data
        """
        if self._client:
            raise ValueError(f"Cannot add new client, client already exists: {self._client}")
        self._client = client
        self._add_to_server()

    def _add_to_server(self) -> None:
        if self._client:
            self._client.add_plot(self._plot_type, self._name, self._id)

    @abstractmethod
    def update(self, *args) -> None:
        """
        Update the tracker's metrics.
        """
        pass

    @abstractmethod
    def get_all_tracked(self, as_np=False):
        """ Retrieve all collected metrics.

        Args:
            as_np (bool): whether to return values as `numpy` arrays

        Returns:
            Tuple: all tracked metrics (size varies by sub-class)
        """
        pass


class TrainValLossTracker(Tracker):
    """
    A tracker object that keeps a record of a model's train and validation loss for a given
    list of steps.
    """
    def __init__(self, name: str, client: Optional[Client] = None):
        """
        Args:
            name (str): the name of this tracker, e.g. "model 1 loss"
            client (Client or None): the client that the tracker is connected to
        """
        super(TrainValLossTracker, self).__init__(name=name, client=client, plot_type=PlotType.train_val_loss)
        self._train: List[float] = []
        self._val: List[float] = []
        self._steps: List[int] = []

        self._add_to_server()

    def get_train_losses(self, as_np=False) -> Union[List, NDArray]:
        """ Retrieve the collected training set losses.
        
        Args:
            as_np (bool): whether to return as a `numpy` array

        Returns:
            List or Array: collected training losses
        """
        if as_np:
            return np.array(self._train)
        return self._train

    def get_val_losses(self, as_np=False) -> Union[List, NDArray]:
        """ Retrieve the collected validation set losses.

        Args:
            as_np (bool): whether to return as a `numpy` array
        
        Returns:
            List or NDArray: collected validation losses
        """
        if as_np:
            return np.array(self._val)
        return self._val

    def get_steps(self, as_np=False) -> Union[List, NDArray]:
        """ Retrieve the collected step numbers.

        Args:
            as_np (bool): whether to return as a `numpy` array
        
        Returns:
            List or NDArray: steps (we do not assume a regular sequence, so this is necessary)
        """
        if as_np:
            return np.array(self._steps)
        return self._steps

    def get_all_tracked(self, as_np=False) -> Tuple[Union[List, NDArray], Union[List, NDArray], Union[List, NDArray]]:
        """ Retrieve all collected metrics.

        Args:
            as_np (bool): whether to return values as `numpy` arrays
        
        Returns:
            Tuple: a 3-tuple of all collected metrics
        """
        if as_np:
            return np.array(self._train), np.array(self._val), np.array(self._steps)
        return self._train, self._val, self._steps

    def update(self, train_loss: float, val_loss: float, step: int) -> None:
        """ Update the tracker's metrics.

        Args:
            train_loss (float): train set loss
            val_loss (float): validation set loss
            step (int): step for which metrics are being gathered
        """
        self._train.append(train_loss)
        self._val.append(val_loss)
        self._steps.append(step)

        if self._client:
            new_data: NDArray = np.array([train_loss, val_loss, step], dtype=np.float32)
            self._client.update_plot(self._id, new_data)


class AccuracyTracker(Tracker):
    """
    A tracker object that keeps a record of a model's accuracies for *categorical* data.
    """
    def __init__(self, name: str, client: Optional[Client] = None):
        """
        Args: 
            name (str): the name of this tracker, e.g. "model 1 loss"
            client (Client or None): the client that the tracker is connected to
        """
        super(AccuracyTracker, self).__init__(name=name, plot_type=PlotType.accuracy)
        self._accuracy: List[float] = []
        self._steps: List[int] = []

        self._add_to_server()
    
    def get_accuracies(self, as_np=False) -> Union[List, NDArray]:
        """ Retrieve the collected accuracies.
        
        Args:
            as_np (bool): whether to return values as `numpy` arrays

        Returns:
            List or NDArray: collected accuracies
        """
        if as_np:
            return np.array(self._accuracy)
        return self._accuracy

    def get_steps(self, as_np=False) -> Union[List, NDArray]:
        """ Retrieve the collected step numbers.

        Args:
            as_np (bool): whether to return values as `numpy` arrays
        
        Returns:
            List or NDarray: steps (we do not assume a regular sequence, so this is necessary)
        """
        if as_np:
            return np.array(self._steps)
        return self._steps

    def get_all_tracked(self, as_np=False) -> Tuple[Union[List, NDArray], Union[List, NDArray]]:
        """ Retrieve all collected metrics.

        Args:
            as_np (bool): whether to return values as `numpy` arrays
        
        Returns:
            Tuple: a 2-tuple of all collected metrics
        """
        if as_np:
            return np.array(self._accuracy), np.array(self._steps)
        return self._accuracy, self._steps

    def update(self, predicted: NDArray, labels: NDArray, step: int) -> None:
        """ Update the tracker's metrics.

        Args:
            predicted (NDArray): predictions (categorical)
            labels (NDArray): ground truth labels (categorical)
            step (int): step for which metrics are being gathered
        """
        n: int = len(labels)
        acc = np.sum(predicted == labels) / n
        self._accuracy.append(acc)

        if self._client:
            new_data: NDArray = np.array([acc, step], dtype=np.float32)
            self._client.update_plot(self._id, new_data)
