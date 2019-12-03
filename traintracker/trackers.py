from abc import ABC, abstractmethod

from traintracker.util.defs import *
from traintracker.client import Client


class Tracker(ABC):
    """
    Base class for all trackers. Trackers are utilities that track various metrics
    regarding the performance of a model.
    """
    def __init__(self, plot_type: PlotType, name: str, client: Optional[Client] = None):
        """
        :param plot_type: type of plot that will be made by server if tracker is connected
            to a client.
        :param name: the name of this tracker, e.g. "model 1 loss"
        :param client: the client that the tracker is connected to
        """
        self._plot_type: PlotType = plot_type
        self._client: Optional[Client] = client
        self._name: str = name

    @property
    def plot_type(self) -> PlotType:
        return self._plot_type

    def connect_client(self, client: Client) -> None:
        """
        Connect this tracker to a client.

        :param client: the client to which this tracker will send its data
        """
        if self._client:
            raise ValueError(f"Cannot add new client, client already exists: {self._client}")
        self._client = client
        self._add_to_server()

    def _add_to_server(self) -> None:
        if self._client:
            self._client.add_plot(self._plot_type, self._name)

    @abstractmethod
    def update(self, *args) -> None:
        """
        Update the tracker's metrics.
        """
        pass

    @abstractmethod
    def get_all_tracked(self, as_np=False):
        """
        Retrieve all collected metrics.

        :param as_np: whether to return values as `numpy` arrays
        """
        pass


class TrainValLossTracker(Tracker):
    """
    A tracker object that keeps a record of a model's train and validation loss for a given
    list of steps.
    """
    def __init__(self, name: str, client: Optional[Client] = None):
        """
        :param name: the name of this tracker, e.g. "model 1 loss"
        :param client: the client that the tracker is connected to
        """
        super(TrainValLossTracker, self).__init__(name=name, client=client, plot_type=PlotType.train_val_loss)
        self._train: List[float] = []
        self._val: List[float] = []
        self._steps: List[int] = []

        self._add_to_server()

    def get_train_losses(self, as_np=False) -> Union[List, NDArray]:
        """
        Retrieve the collected training set losses.

        :param as_np: whether to return as a `numpy` array
        :return: a list or array of collected training losses
        """
        if as_np:
            return np.array(self._train)
        return self._train

    def get_val_losses(self, as_np=False) -> Union[List, NDArray]:
        """
        Retrieve the collected validation set losses.

        :param as_np: whether to return as a `numpy` array
        :return: a list or array of collected validation losses
        """
        if as_np:
            return np.array(self._val)
        return self._val

    def get_steps(self, as_np=False) -> Union[List, NDArray]:
        """
        Retrieve the collected step numbers.

        :param as_np: whether to return as a `numpy` array
        :return: a list or array of steps
            (we do not assume a regular sequence, so this is necessary)
        """
        if as_np:
            return np.array(self._steps)
        return self._steps

    def get_all_tracked(self, as_np=False) -> Tuple[Union[List, NDArray], Union[List, NDArray], Union[List, NDArray]]:
        """
        Retrieve all collected metrics.

        :param as_np: whether to return values as `numpy` arrays
        :return: a 3-tuple of all collected metrics
        """
        if as_np:
            return np.array(self._train), np.array(self._val), np.array(self._steps)
        return self._train, self._val, self._steps

    def update(self, train_loss: float, val_loss: float, step: int) -> None:
        """
        Update the tracker's metrics.

        :param train_loss: train set loss
        :param val_loss: validation set loss
        :param step: step for which metrics are being gathered
        """
        self._train.append(train_loss)
        self._val.append(val_loss)
        self._steps.append(step)

        if self._client:
            new_data: NDArray = np.array([train_loss, val_loss, step], dtype=np.float32)
            self._client.update_plot(self._name, new_data)


class AccuracyTracker(Tracker):
    def __init__(self, name: str, client: Optional[Client] = None):
        """
        :param name: the name of this tracker, e.g. "model 1 loss"
        :param client: the client that the tracker is connected to
        """
        super(AccuracyTracker, self).__init__(name=name, plot_type=PlotType.accuracy)
        self._accuracy: List[float] = []
        self._steps: List[int] = []
    
    def get_accuracies(self, as_np=False) -> Union[List, NDArray]:
        """
        Retrieve the collected accuracies.

        :param as_np: whether to return values as `numpy` arrays
        :return: a list or array of collected accuracies
        """
        if as_np:
            return np.array(self._accuracy)
        return self._accuracy

    def get_steps(self, as_np=False) -> Union[List, NDArray]:
        """
        Retrieve the collected step numbers.

        :param as_np: whether to return values as `numpy` arrays
        :return: a list or array of steps
            (we do not assume a regular sequence, so this is necessary)
        """
        if as_np:
            return np.array(self._steps)
        return self._steps

    def get_all_tracked(self, as_np=False) -> Tuple[Union[List, NDArray], Union[List, NDArray]]:
        """
        Retrieve all collected metrics.

        :param as_np: whether to return values as `numpy` arrays
        :return: a 2-tuple of all collected metrics
        """
        if as_np:
            return np.array(self._accuracy), np.array(self._steps)
        return self._accuracy, self._steps

    def update(self, predicted: NDArray, labels: NDArray, step: int) -> None:
        """
        Update the tracker's metrics.

        :param predicted: predictions (categorical)
        :param labels: ground truth labels (categorical)
        :param step: step for which metrics are being gathered
        """
        n: int = len(labels)
        acc = np.sum(predicted == labels) / n
        self._accuracy.append(acc)

        if self._client:
            new_data: NDArray = np.array([acc, step], dtype=np.float32)
            self._client.update_plot(self._name, new_data)
