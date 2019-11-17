import socket
import sys
from bokeh.server.server import Server
from bokeh.application import Application
from bokeh.application.handlers.function import FunctionHandler
from bokeh.plotting import figure, ColumnDataSource
from bokeh.document.document import Document
from bokeh.plotting.figure import Figure
import random

from train_tracker.util.defs import *

PS_PORT = 12345
TIMEOUT = 100


class TrainTracker:
    _sources: Dict[PlotType, Dict] = {
        PlotType.random: {'x': [], 'y': []},
    }

    def __init__(self):
        # self._apps: Dict[str, Application] = {"/": Application(FunctionHandler(self.make_document))}
        self._plot_server: Optional[Server] = None
        self._port: int = PS_PORT
        self._sources: Dict[PlotType, ColumnDataSource] = {}
        self._plots: Dict[PlotType, Figure] = {}

    def add_plot(self, plot_type):
        if plot_type not in self._sources:
            self._sources[plot_type] = ColumnDataSource(self._sources[plot_type])
            self._plots[plot_type] = self._build_plot(plot_type)

    def make_document(self, doc: Document):
        doc.title = "Testing..."
        for _, plot in self._plots.items():
            doc.add_root(plot)

        doc.add_periodic_callback(self.update, TIMEOUT)

    def update(self) -> None:
        if PlotType.random in self._plots:
            new = {'x': [random.random()], 'y': [random.random()]}
            self._sources[PlotType.random].stream(new)

    def start(self) -> None:
        # self._server = Server(self._apps, self._port)
        self._plot_server = Server({'/': self.make_document}, port=PS_PORT, num_procs=1)
        self._plot_server.start()
        self._plot_server.io_loop.add_callback(self._plot_server.show, "/")
        self._plot_server.io_loop.start()

    def _build_plot(self, plot_type: PlotType) -> Figure:
        if plot_type == PlotType.random:
            fig = figure(title="Random")
            fig.circle(source=self._sources[plot_type], x='x', y='y', size=10)
        else:
            raise ValueError(f"Bad plot type argument ({plot_type.name})")
        return fig

