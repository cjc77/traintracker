import socket

from train_tracker.util.defs import *

FAIL_MSG = "Correct {} not received by server"


class Client:
    def __init__(self, host: str, port: int):
        self._host = host
        self._port = port
        self._socket: Optional[socket.socket] = None

    def connect(self) -> None:
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._socket.connect((self._host, self._port))

    def close_connection(self) -> None:
        self._socket.close()
        self._socket = None

    def add_plot(self, plot_type: PlotType) -> None:
        self.send_cmd(Cmd.add_plot)
        self._socket.sendall(plot_type.to_bytes(INT32, BYTEORDER))
        ack = self._socket.recv(BUFFSIZE)
        ack = int.from_bytes(ack, BYTEORDER)
        assert ack == plot_type, FAIL_MSG.format("plot type")

    def start_plot_server(self) -> None:
        self.send_cmd(Cmd.start_plot_server)

    def shutdown_server(self) -> None:
        self.send_cmd(Cmd.server_shutdown)

    def send_cmd(self, cmd: Cmd) -> None:
        if self._socket:
            self._socket.sendall(cmd.to_bytes(INT32, BYTEORDER))
            ack = self._socket.recv(BUFFSIZE)
            ack = int.from_bytes(ack, BYTEORDER)
            assert ack == cmd, FAIL_MSG.format("command")
