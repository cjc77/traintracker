import socket
import numpy as np
from abc import ABC, abstractmethod

from traintracker.util.defs import *

FAIL_MSG = "Correct data not received by server, received: {}, expected: {}"
FAIL_SPEC = "Point of failure: {}"


class Client:
    """ A client is responsible for passing along data to the server. 

    A client will be referenced by all trackers that wish to send data 
    to the server.
    """
    def __init__(self):
        self._host: Optional[str] = None
        self._port: Optional[int] = None
        self._socket: Optional[socket.socket] = None

    def connect(self, host: str, port: int) -> None:
        """ Connect client to a server.

        Args:
            host (str): host where server is running
            port (int): port where server is listening
        """
        self._host = host
        self._port = port
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._socket.connect((self._host, self._port))

    def close_connection(self) -> None:
        """
        Close connection with the server.
        """
        self._socket.close()
        self._socket = None

    def add_plot(self, plot_type: PlotType, plot_name: str, tracker_id: int) -> None:
        self._send_cmd(Cmd.add_plot)
        data: bytes = plot_type.to_bytes(INT32, BYTEORDER)
        self._safe_send(data)
        data: bytes = tracker_id.to_bytes(INT32, BYTEORDER)
        self._safe_send(data)
        data = plot_name.encode()
        self._safe_send(len(data).to_bytes(INT32, BYTEORDER))
        self._safe_send(data)

    def update_plot(self, plot_id: int, new_data: NDArray) -> None:
        self._send_cmd(Cmd.update_plot)
        # data = plot_name.encode()
        data: bytes = plot_id.to_bytes(INT32, BYTEORDER)
        self._safe_send(len(data).to_bytes(INT32, BYTEORDER))
        self._safe_send(data)

        if new_data.dtype != np.float32:
            new_data: np.array = np.array(new_data, dtype=np.float32)
        data = new_data.tobytes()
        self._safe_send(len(data).to_bytes(INT32, BYTEORDER))
        self._safe_send(data)

    def start_plot_server(self) -> None:
        """
        Instruct the server to start the plot server.
        """
        self._send_cmd(Cmd.start_plot_server)

    def shutdown_server(self) -> None:
        """
        Instruct server to shutdown (that we are done with it)
        """
        self._send_cmd(Cmd.server_shutdown)

    def _send_cmd(self, cmd: Cmd) -> None:
        data: bytes = cmd.to_bytes(INT32, BYTEORDER)
        self._safe_send(data)

    def _safe_send(self, data: bytes) -> None:
        if self._socket:
            self._socket.sendall(data)
