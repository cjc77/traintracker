import asyncio
from asyncio import StreamReader, StreamWriter
from queue import Queue
import numpy as np
from abc import ABC, abstractmethod
from bokeh.server.server import Server as BokehServer
from bokeh.plotting import figure, ColumnDataSource
from bokeh.document.document import Document
from bokeh.plotting.figure import Figure

from train_tracker.util.defs import *


class TrackerPlot(ABC):
    def __init__(self, name: str, source: ColumnDataSource):
        self._name: str = name
        self.fig: Optional[Figure] = None

        self.source: ColumnDataSource = source

    @classmethod
    def build_plot(cls, plot_type: PlotType, name: str, source: ColumnDataSource):
        if plot_type == PlotType.train_val_loss:
            return TrainValLossPlot(name, source)

    @abstractmethod
    def update(self, new_data) -> None:
        pass


class TrainValLossPlot(TrackerPlot):
    def __init__(self, name: str, source: ColumnDataSource):
        super(TrainValLossPlot, self).__init__(name=name, source=source)
        self._init_figure()

    def update(self, new_data) -> None:
        new = {"train": [new_data[0]], "val": [new_data[1]], "epoch": [new_data[2]]}
        self.source.stream(new)

    def _init_figure(self) -> None:
        self.fig = figure(title=self._name)
        self.fig.line(source=self.source, x="epoch", y="train", color="blue", legend="training loss")
        self.fig.line(source=self.source, x="epoch", y="val", color="orange", legend="validation loss")
        self.fig.xaxis.axis_label = "Epoch"
        self.fig.yaxis.axis_label = "Loss"


class Server:
    _source_formats: Dict[PlotType, Dict] = {
        PlotType.train_val_loss: {"train": [], "val": [], "epoch": []},
        PlotType.random: {'x': [], 'y': []},
        PlotType.test_line_plt: {'x': [], 'y': []}
    }

    def __init__(self):
        self._host: Optional[str] = None
        self._port: Optional[int] = None
        self._plot_server_port: Optional[int] = None

        self._reader: Optional[StreamReader] = None
        self._writer: Optional[StreamWriter] = None
        self._plot_server: Optional[BokehServer] = None

        self._sources: Dict[str, ColumnDataSource] = {}
        self._plots: Dict[str, TrackerPlot] = {}
        self._queues: Dict[str, Queue] = {}

    def run(self, host: str, port: int = PORT, plots_port: int = PS_PORT) -> None:
        self._host = host
        self._port = port
        self._plot_server_port = plots_port
        try:
            asyncio.run(self._run_async())
        except RuntimeError as re:
            print(f"Server shutdown with runtime error: {re}")

    async def _run_async(self) -> None:
        server = await asyncio.start_server(self._handle_serving, self._host, self. _port)
        addr: Tuple[str, int] = server.sockets[0].getsockname()
        print(f"Serving at {addr[0]} on port {addr[1]}")
        async with server:
            await server.serve_forever()

    def _start_plot_server(self) -> None:
        self._plot_server = BokehServer({'/': self._make_document}, port=self._plot_server_port, num_procs=1)
        self._plot_server.start()
        self._plot_server.io_loop.add_callback(self._plot_server.show, "/")
        print(f"Serving plots on port: {self._plot_server_port}")
        # self._plot_server.io_loop.start()

    def _make_document(self, doc: Document) -> None:
        doc.title = "Testing..."
        for _, plot in self._plots.items():
            doc.add_root(plot.fig)

        doc.add_periodic_callback(lambda: self._update_plots(doc), TIMEOUT)

    async def _handle_serving(self, reader: StreamReader, writer: StreamWriter) -> None:
        self._writer = writer
        self._reader = reader

        while True:
            cmd = await self._reader.read(BUFFSIZE)
            cmd = int.from_bytes(cmd, BYTEORDER)
            print(f"Received command: {Cmd(cmd).name}")

            # Send Ack
            await self._write_and_drain(cmd.to_bytes(INT32, BYTEORDER))

            # If command is server_shutdown, this is a special case
            if cmd == Cmd.server_shutdown:
                print(f"Closing connection...")
                self._writer.close()
                asyncio.get_event_loop().stop()
                return

            # Handle command
            await self._handle_cmd(Cmd(cmd))

    async def _handle_cmd(self, cmd: Cmd) -> None:
        if cmd == Cmd.update_plot:
            plot_name: bytes = await self._reader.read(BUFFSIZE)
            plot_name: str = plot_name.decode()
            print(f"Updating: {plot_name}")
            await self._write_and_drain(plot_name.encode())
            await self._handle_plot_update(plot_name)
        elif cmd == Cmd.add_plot:
            plot_type = await self._reader.read(BUFFSIZE)
            plot_type = PlotType(int.from_bytes(plot_type, BYTEORDER))
            print(f"Plots: {self._plots}")
            await self._write_and_drain(plot_type.to_bytes(INT32, BYTEORDER))
            plot_name: bytes = await self._reader.read(BUFFSIZE)
            plot_name: str = plot_name.decode()
            print(f"Adding: {plot_name}")
            await self._write_and_drain(plot_name.encode())
            self._add_plot(plot_type, plot_name)
            print(f"Plots: {self._plots}")
        elif cmd == Cmd.start_plot_server:
            self._start_plot_server()

    async def _handle_plot_update(self, plot_name: str) -> None:
        new_data: NDArray = np.frombuffer(await self._reader.read(BUFFSIZE), dtype=np.float32)
        print(f"Received data: {new_data}")
        await self._write_and_drain(len(new_data).to_bytes(INT32, BYTEORDER))
        self._queues[plot_name].put(new_data)

    async def _write_and_drain(self, data: bytes) -> None:
        self._writer.write(data)
        await self._writer.drain()

    def _add_plot(self, plot_type: PlotType, plot_name: str) -> None:
        if plot_name not in self._plots:
            src = ColumnDataSource(self._source_formats[plot_type])
            self._plots[plot_name] = TrackerPlot.build_plot(plot_type, plot_name, src)
            self._queues[plot_name] = Queue()

    def _update_plots(self, doc: Document) -> None:
        for name, plot in self._plots.items():
            q = self._queues[name]
            while not q.empty():
                plot.update(q.get())
