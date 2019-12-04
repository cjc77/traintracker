import asyncio
from asyncio import StreamReader, StreamWriter
from queue import Queue
from bokeh.server.server import Server as BokehServer
from bokeh.plotting import figure, ColumnDataSource, gridplot
from bokeh.document.document import Document
from copy import deepcopy
from dask import delayed, compute

from traintracker.util.defs import *
from traintracker.tracker_plots import TrackerPlot


class Server:
    """ A server is responsible for communicating with the client about plots.
    
    Communications involves commands regarding plot creation and updating.
    It is also responsible for managing a separate plot server.
    """
    def __init__(self):
        self._host: Optional[str] = None
        self._port: Optional[int] = None
        self._plot_server_port: Optional[int] = None

        self._reader: Optional[StreamReader] = None
        self._writer: Optional[StreamWriter] = None
        self._plot_server: Optional[BokehServer] = None

        self._plots: Dict[str, TrackerPlot] = {}
        self._queues: Dict[str, Queue] = {}

    def run(self, host: str, port: int = PORT, plots_port: int = PS_PORT) -> None:
        """ Run the server.

        Args:
            host (str): host on which to run
            port (int): port on which to run the server
            plots_port (int): port on which to serve plots
        """
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
        doc.title = "Train Tracker"
        # for _, plot in self._plots.items():
        #     doc.add_root(plot.fig)
        figs = [plot.fig for _, plot in self._plots.items()]
        grid = [figs[i: i + 3] for i in range(0, len(figs), 3)]
        doc.add_root(gridplot(grid))

        doc.add_periodic_callback(lambda: self._update_plots(doc), TIMEOUT)

    async def _handle_serving(self, reader: StreamReader, writer: StreamWriter) -> None:
        self._writer = writer
        self._reader = reader

        while True:
            cmd = await self._reader.read(INT32)
            cmd = int.from_bytes(cmd, BYTEORDER)
            # print(f"Received command: {Cmd(cmd).name}")

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
            plot_name_size: int = int.from_bytes(await self._reader.read(INT32), BYTEORDER)
            plot_name: bytes = await self._reader.read(plot_name_size)
            plot_name: str = plot_name.decode()
            # print(f"Updating: {plot_name}")
            await self._handle_plot_update(plot_name)
        elif cmd == Cmd.add_plot:
            plot_type = await self._reader.read(INT32)
            plot_type = PlotType(int.from_bytes(plot_type, BYTEORDER))
            plot_name_size: int = int.from_bytes(await self._reader.read(INT32), BYTEORDER)
            plot_name: bytes = await self._reader.read(plot_name_size)
            plot_name: str = plot_name.decode()
            # print(f"Adding: {plot_name}")
            self._add_plot(plot_type, plot_name)
        elif cmd == Cmd.start_plot_server:
            self._start_plot_server()

    async def _handle_plot_update(self, plot_name: str) -> None:
        new_data_size: int = int.from_bytes(await self._reader.read(INT32), BYTEORDER)
        new_data: NDArray = np.frombuffer(await self._reader.read(new_data_size),
                                          dtype=np.float32)
        self._queues[plot_name].put(new_data)

    async def _write_and_drain(self, data: bytes) -> None:
        self._writer.write(data)
        await self._writer.drain()

    def _add_plot(self, plot_type: PlotType, plot_name: str) -> None:
        if plot_name not in self._plots:
            self._plots[plot_name] = TrackerPlot.build_plot(plot_type, plot_name)
            self._queues[plot_name] = Queue()

    def _update_plots(self, doc: Document) -> None:
        # update each plot/queue pair in parallel
        compute(
            delayed(plot.update_from_queue)(self._queues[name], doc) for name, plot, in self._plots.items()
        )

