import asyncio
from asyncio import StreamReader, StreamWriter
import random
from queue import Queue
import numpy as np
from bokeh.server.server import Server as BokehServer
from bokeh.plotting import figure, ColumnDataSource
from bokeh.document.document import Document
from bokeh.plotting.figure import Figure

from train_tracker.util.defs import *


class Server:
    _source_formats: Dict[PlotType, Dict] = {
        PlotType.random: {'x': [], 'y': []},
        PlotType.test_line_plt: {'x': [], 'y': []}
    }

    def __init__(self, host: str, port: int):
        self._host = host
        self._port = port
        self._reader: Optional[StreamReader]
        self._writer: Optional[StreamWriter]
        self._plot_server: Optional[BokehServer] = None
        self._plot_server_port: int = PS_PORT
        self._sources: Dict[PlotType, ColumnDataSource] = {}
        self._plots: Dict[PlotType, Figure] = {}
        self._queues: Dict[PlotType, Queue] = {}

    def run(self) -> None:
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
        # self._plot_server.io_loop.start()

    def _make_document(self, doc: Document) -> None:
        doc.title = "Testing..."
        for _, plot in self._plots.items():
            doc.add_root(plot)

        doc.add_periodic_callback(self._update_plots, TIMEOUT)

    async def _handle_serving(self, reader: StreamReader, writer: StreamWriter) -> None:
        self._writer = writer
        self._reader = reader

        while True:
            cmd = await self._reader.read(BUFFSIZE)
            cmd = int.from_bytes(cmd, BYTEORDER)
            print(f"Received: {cmd}")

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
        print(f"Command: {cmd.name}")
        if cmd == Cmd.update_plot:
            plot_type = await self._reader.read(BUFFSIZE)
            plot_type = PlotType(int.from_bytes(plot_type, BYTEORDER))
            print(f"Updating: {plot_type.name}")
            await self._write_and_drain(plot_type.to_bytes(INT32, BYTEORDER))
            await self._handle_plot_update(plot_type)
        elif cmd == Cmd.add_plot:
            plot_type = await self._reader.read(BUFFSIZE)
            plot_type = PlotType(int.from_bytes(plot_type, BYTEORDER))
            self._add_plot(plot_type)
            print(f"Plots: {self._plots}")
            await self._write_and_drain(plot_type.to_bytes(INT32, BYTEORDER))
        elif cmd == Cmd.start_plot_server:
            self._start_plot_server()

    async def _handle_plot_update(self, plot_type: PlotType) -> None:
        if plot_type == PlotType.test_line_plt:
            new_data: NDArray = np.frombuffer(await self._reader.read(BUFFSIZE), dtype=np.float32)
            print(f"Received data: {new_data}")
            await self._write_and_drain(len(new_data).to_bytes(INT32, BYTEORDER))
            self._queues[plot_type].put(new_data)

    async def _write_and_drain(self, data: bytes) -> None:
        self._writer.write(data)
        await self._writer.drain()

    def _build_plot(self, plot_type: PlotType) -> Figure:
        if plot_type == PlotType.random:
            fig = figure(title="Random")
            fig.circle(source=self._sources[plot_type], x='x', y='y', size=10)
        elif plot_type == PlotType.test_line_plt:
            fig = figure(title="Test Line")
            fig.line(source=self._sources[plot_type], x='x', y='y')
        else:
            raise ValueError(f"Bad plot type argument ({plot_type.name})")
        return fig

    def _add_plot(self, plot_type) -> None:
        if plot_type not in self._sources:
            self._sources[plot_type] = ColumnDataSource(self._source_formats[plot_type])
            self._plots[plot_type] = self._build_plot(plot_type)
            self._queues[plot_type] = Queue()

    def _update_plots(self) -> None:
        if PlotType.random in self._plots:
            new = {'x': [random.random()], 'y': [random.random()]}
            self._sources[PlotType.random].stream(new)
        if PlotType.test_line_plt in self._plots:
            q = self._queues[PlotType.test_line_plt]
            while not q.empty():
                new_data = self._queues[PlotType.test_line_plt].get()
                new = {'x': [new_data[0]], 'y': [new_data[1]]}
                self._sources[PlotType.test_line_plt].stream(new)
