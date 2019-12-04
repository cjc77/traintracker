from abc import ABC, abstractmethod
from queue import Queue
from bokeh.document.document import Document
from bokeh.plotting import figure, ColumnDataSource
from bokeh.plotting.figure import Figure
from copy import deepcopy

from traintracker.util.defs import *


SOURCE_FORMATS: Dict[PlotType, Dict] = {
    PlotType.train_val_loss: {"train": [], "val": [], "step": []},
    PlotType.accuracy: {"acc": [], "step": []},
    PlotType.random: {'x': [], 'y': []},
    PlotType.test_line_plt: {'x': [], 'y': []}
}


class TrackerPlot(ABC):
    """
    A plot that corresponds to a tracker on the client side.
    """
    def __init__(self, name: str, id_: int, source: ColumnDataSource):
        """ A plot that corresponds with a tracker.
        
        Args:
            name (str): name of this plot
            id_ (int): a unique id that identifies both this plot and the tracker
                that is related to it.
            source (ColumnDataSource): a columnar data source from which this plot 
                receives updates
        """
        self._name: str = name
        self._id: int = id_
        self.fig: Optional[Figure] = None

        self.source: ColumnDataSource = source

    @classmethod
    def build_plot(cls, plot_type: PlotType, name: str, id_: int) -> "TrackerPlot":
        """
        Args:
            plot_type (PlotType): type of plot to be created
            name (str): name of plot to be created
            id_ (int): a unique id that identifies both this plot and the tracker
                that is related to it.
        """
        source = ColumnDataSource(deepcopy(SOURCE_FORMATS[plot_type]))
        if plot_type == PlotType.train_val_loss:
            return TrainValLossPlot(name, id_, source)
        elif plot_type == PlotType.accuracy:
            return AccuraccyPlot(name, id_, source)

    @property
    def id(self) -> int:
        return self._id

    @abstractmethod
    def update(self, new_data, doc: Document) -> None:
        pass

    @abstractmethod
    def update_from_queue(self, new_data_queue: Queue, doc: Document) -> None:
        pass


class TrainValLossPlot(TrackerPlot):
    def __init__(self, name: str, id_: int, source: ColumnDataSource):
        super(TrainValLossPlot, self).__init__(name=name, id_=id_, source=source)
        self._init_figure()

    def update(self, new_data, doc: Document) -> None:
        new = {"train": [new_data[0]], "val": [new_data[1]], "step": [new_data[2]]}
        # add_next_tick_callback() can be used safely without taking the document lock
        doc.add_next_tick_callback(lambda: self.source.stream(new))

    def update_from_queue(self, new_data_queue: Queue, doc: Document) -> None:
        while not new_data_queue.empty():
            new_data = new_data_queue.get_nowait()
            # add_next_tick_callback() can be used safely without taking the document lock
            doc.add_next_tick_callback(
                lambda: self.source.stream({"train": [new_data[0]],
                                            "val": [new_data[1]],
                                            "step": [new_data[2]]})
            )

    def _init_figure(self) -> None:
        self.fig = figure(title=self._name)
        self.fig.line(source=self.source, x="step", y="train", color="blue", legend="training loss")
        self.fig.line(source=self.source, x="step", y="val", color="orange", legend="validation loss")
        self.fig.xaxis.axis_label = "Step"
        self.fig.yaxis.axis_label = "Loss"


class AccuraccyPlot(TrackerPlot):
    def __init__(self, name: str, id_: int, source: ColumnDataSource):
        super(AccuraccyPlot, self).__init__(name=name, id_=id_, source=source)
        self._init_figure()

    def update(self, new_data, doc: Document) -> None:
        new = {"acc": [new_data[0]], "step": [new_data[1]]}
        # add_next_tick_callback() can be used safely without taking the document lock
        doc.add_next_tick_callback(lambda: self.source.stream(new))

    def update_from_queue(self, new_data_queue: Queue, doc: Document) -> None:
        while not new_data_queue.empty():
            new_data = new_data_queue.get_nowait()
            # add_next_tick_callback() can be used safely without taking the document lock
            doc.add_next_tick_callback(
                lambda: self.source.stream({"acc": [new_data[0]], "step": [new_data[1]]})
            )

    def _init_figure(self) -> None:
        self.fig = figure(title=self._name)
        self.fig.line(source=self.source, x="step", y="acc", color="blue", legend="accuracy")
        self.fig.xaxis.axis_label = "Step"
        self.fig.yaxis.axis_label = "Accuracy"
