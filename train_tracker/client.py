import socket
import numpy as np
from abc import ABC, abstractmethod

from train_tracker.util.defs import *

FAIL_MSG = "Correct data not received by server, received: {}, expected: {}"
FAIL_SPEC = "Point of failure: {}"


class Client:
    def __init__(self):
        self._host: Optional[str] = None
        self._port: Optional[int] = None
        self._socket: Optional[socket.socket] = None

    def connect(self, host: str, port: int) -> None:
        self._host = host
        self._port = port
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._socket.connect((self._host, self._port))

    def close_connection(self) -> None:
        self._socket.close()
        self._socket = None

    def add_plot(self, plot_type: PlotType) -> None:
        self._send_cmd(Cmd.add_plot)
        data: bytes = plot_type.to_bytes(INT32, BYTEORDER)
        self._safe_send(data, assertion=True, expected=plot_type, fail_msg=FAIL_MSG)

    def update_plot(self, plot_type: PlotType, new_data: NDArray) -> None:
        self._send_cmd(Cmd.update_plot)
        data: bytes = plot_type.to_bytes(INT32, BYTEORDER)
        self._safe_send(data, assertion=True, expected=plot_type, fail_msg=FAIL_MSG)

        if new_data.dtype != np.float32:
            new_data: np.array = np.array(new_data, dtype=np.float32)
        data = new_data.tobytes()
        print(f"Sending data: {new_data}")
        self._safe_send(data, assertion=True, expected=len(new_data), fail_msg=FAIL_MSG)

    def start_plot_server(self) -> None:
        self._send_cmd(Cmd.start_plot_server)

    def shutdown_server(self) -> None:
        self._send_cmd(Cmd.server_shutdown)

    def _send_cmd(self, cmd: Cmd) -> None:
        data: bytes = cmd.to_bytes(INT32, BYTEORDER)
        self._safe_send(data, assertion=True, expected=cmd, fail_msg=FAIL_MSG)

    def _safe_send(self, data: bytes, assertion=False, expected: int = 0, fail_msg="", fail_spec="", buffsize=BUFFSIZE):
        if self._socket:
            self._socket.sendall(data)
            ack = self._socket.recv(buffsize)
            ack = int.from_bytes(ack, BYTEORDER)
            if assertion:
                assert ack == expected, fail_msg.format(ack, expected) + fail_spec


class Tracker(ABC):
    def __init__(self, client: Client, plot_type: PlotType):
        self._client: Client = client
        self._plot_type: PlotType = plot_type

    @property
    def plot_type(self) -> PlotType:
        return self._plot_type

    def _add_to_server(self) -> None:
        self._client.add_plot(self._plot_type)

    @abstractmethod
    def update(self, *args) -> None:
        pass


class RandomTracker(Tracker):
    def __init__(self, client: Client):
        super(RandomTracker, self).__init__(client, plot_type=PlotType.random)
        self._x: List[float] = []
        self._y: List[float] = []

        self._add_to_server()

    def update(self) -> None:
        new_data: NDArray = np.random.random(2)
        self._x.append(new_data[0])
        self._y.append(new_data[1])
        self._client.update_plot(self._plot_type, new_data)


class TestLineTracker(Tracker):
    def __init__(self, client: Client):
        super(TestLineTracker, self).__init__(client, plot_type=PlotType.test_line_plt)
        self._x: List[float] = []
        self._y: List[float] = []

        self._add_to_server()

    def update(self, x: float, y: float) -> None:
        new_data: NDArray = np.array([x, y], dtype=np.float32)
        self._x.append(new_data[0])
        self._y.append(new_data[1])
        self._client.update_plot(self._plot_type, new_data)

