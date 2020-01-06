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
    PlotType.conf_mtx: {"pred": [], "true": []},
    PlotType.random: {'x': [], 'y': []},
    PlotType.test_line_plt: {'x': [], 'y': []}
}

class TrackerPlot(ABC):
    """
    A plot that corresponds to a tracker on the client side.
    """
    def __init__(self, name: str, id_: int, plot_type: PlotType, source: ColumnDataSource):
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
        self.type: PlotType = plot_type
        self.fig: Optional[Figure] = None

        self.source: ColumnDataSource = source

    @classmethod
    def build_plot(cls, plot_type: PlotType, name: str, id_: int, *args) -> "TrackerPlot":
        """
        Args:
            plot_type (PlotType): type of plot to be created
            name (str): name of plot to be created
            id_ (int): a unique id that identifies both this plot and the tracker
                that is related to it.
        """
        source = ColumnDataSource(deepcopy(SOURCE_FORMATS[plot_type]))
        if plot_type == PlotType.train_val_loss:
            return TrainValLossPlot(name, id_, plot_type, source)
        elif plot_type == PlotType.accuracy:
            return AccuraccyPlot(name, id_, plot_type, source)
        elif plot_type == PlotType.conf_mtx:
            m = args[0]
            levels = args[1]
            return ConfusionMatrixPlot(name, id_, plot_type, source, m, levels)
        else:
            raise ValueError(f"{PlotType} is not a valid PlotType.")

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
    def __init__(self, name: str, id_: int, plot_type: PlotType, source: ColumnDataSource):
        super(TrainValLossPlot, self).__init__(name=name, id_=id_, plot_type=plot_type, source=source)
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
    def __init__(self, name: str, id_: int, plot_type: PlotType, source: ColumnDataSource):
        super(AccuraccyPlot, self).__init__(name=name, id_=id_, plot_type=plot_type, source=source)
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


class ConfusionMatrixPlot(TrackerPlot):
    def __init__(self, name: str, id_: int, plot_type: PlotType, source: ColumnDataSource, m: int, levels: NDCharArr):
        super(ConfusionMatrixPlot, self).__init__(name=name, id_=id_, plot_type=plot_type, source=source)
        self._m = m
        self.levels = levels
        self.str_levels = [l.decode() for l in self.levels]
        self.item_size = self.levels.itemsize
        print(f"m: {self._m}, levels: {self.levels}")
        self._init_figure()

    def update(self, new_data, doc: Document) -> None:
        pass

    def update_from_queue(self, new_data_queue: Queue, doc: Document) -> None:
        while not new_data_queue.empty():
            new_data = new_data_queue.get_nowait()
            doc.add_next_tick_callback(
                lambda: self.source.stream(
                    {"pred": [d.decode() for d in new_data[0]], 
                     "true": [d.decode() for d in new_data[1]]})
            )

    def _init_figure(self) -> None:
        self.fig = figure(title=self._name, x_range=self.str_levels, y_range=self.str_levels)
        self.fig.xgrid.visible = False
        self.fig.ygrid.visible = False
        self.fig.rect(source=self.source, x="pred", y="true", width=1, height=1)

