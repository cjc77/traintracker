from abc import ABC, abstractmethod
from queue import Queue
from bokeh.document.document import Document
from bokeh.plotting import figure, ColumnDataSource
from bokeh.plotting.figure import Figure

from traintracker.util.defs import *


class TrackerPlot(ABC):
    def __init__(self, name: str, source: ColumnDataSource):
        """
        A plot that corresponds with a tracker.

        :param name: name of this plot
        :param source: a columnar data source from which this plot receives updates
        """
        self._name: str = name
        self.fig: Optional[Figure] = None

        self.source: ColumnDataSource = source

    @classmethod
    def build_plot(cls, plot_type: PlotType, name: str, source: ColumnDataSource) -> "TrackerPlot":
        """
        :param plot_type: type of plot to be created
        :param name: name of plot to be created
        :param source: a columnar data source from which this plot receives updates
        :return: an initialized tracker plot
        """
        if plot_type == PlotType.train_val_loss:
            return TrainValLossPlot(name, source)

    @abstractmethod
    def update(self, new_data, doc: Document) -> None:
        pass

    @abstractmethod
    def update_from_queue(self, new_data_queue: Queue, doc: Document) -> None:
        pass


class TrainValLossPlot(TrackerPlot):
    def __init__(self, name: str, source: ColumnDataSource):
        super(TrainValLossPlot, self).__init__(name=name, source=source)
        self._init_figure()

    def update(self, new_data, doc: Document) -> None:
        new = {"train": [new_data[0]], "val": [new_data[1]], "epoch": [new_data[2]]}
        # add_next_tick_callback() can be used safely without taking the document lock
        doc.add_next_tick_callback(lambda: self.source.stream(new))

    def update_from_queue(self, new_data_queue: Queue, doc: Document) -> None:
        while not new_data_queue.empty():
            new_data = new_data_queue.get_nowait()
            # add_next_tick_callback() can be used safely without taking the document lock
            doc.add_next_tick_callback(
                lambda: self.source.stream({"train": [new_data[0]],
                                            "val": [new_data[1]],
                                            "epoch": [new_data[2]]})
            )

    def _init_figure(self) -> None:
        self.fig = figure(title=self._name)
        self.fig.line(source=self.source, x="epoch", y="train", color="blue", legend="training loss")
        self.fig.line(source=self.source, x="epoch", y="val", color="orange", legend="validation loss")
        self.fig.xaxis.axis_label = "Step"
        self.fig.yaxis.axis_label = "Loss"

