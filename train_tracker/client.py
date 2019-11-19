import socket
import numpy as np
from abc import ABC, abstractmethod

from train_tracker.util.defs import *

FAIL_MSG = "Correct data not received by server, received: {}, expected: {}"
FAIL_SPEC = "Point of failure: {}"


class Client:
    """
    A client is responsible for passing along data to the server. A client will be
    referenced by all trackers that wish to send data to the server.
    """
    def __init__(self):
        self._host: Optional[str] = None
        self._port: Optional[int] = None
        self._socket: Optional[socket.socket] = None

    def connect(self, host: str, port: int) -> None:
        """
        Connect client to a server.

        :param host: host where server is running
        :param port: port where server is listening
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

    def add_plot(self, plot_type: PlotType, plot_name: str) -> None:
        self._send_cmd(Cmd.add_plot)
        data: bytes = plot_type.to_bytes(INT32, BYTEORDER)
        self._safe_send(data, assertion=True, expected=plot_type.value, fail_msg=FAIL_MSG)
        data = plot_name.encode()
        self._safe_send(data, assertion=True, expected=plot_name, fail_msg=FAIL_MSG)

    def update_plot(self, plot_name: str, new_data: NDArray) -> None:
        self._send_cmd(Cmd.update_plot)
        data = plot_name.encode()
        self._safe_send(data, assertion=True, expected=plot_name, fail_msg=FAIL_MSG)

        if new_data.dtype != np.float32:
            new_data: np.array = np.array(new_data, dtype=np.float32)
        data = new_data.tobytes()
        print(f"Sending data: {new_data}")
        self._safe_send(data, assertion=True, expected=len(new_data), fail_msg=FAIL_MSG)

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
        self._safe_send(data, assertion=True, expected=cmd.value, fail_msg=FAIL_MSG)

    def _safe_send(self, data: bytes, assertion=False, expected: Union[int, str] = 0,
                   fail_msg="", fail_spec="", buffsize=BUFFSIZE) -> None:
        if self._socket:
            self._socket.sendall(data)
            ack = self._socket.recv(buffsize)
            if type(expected) == int:
                ack = int.from_bytes(ack, BYTEORDER)
            elif type(expected) == str:
                ack = ack.decode()
            if assertion:
                assert ack == expected, fail_msg.format(ack, expected) + fail_spec


class Tracker(ABC):
    def __init__(self, client: Client, plot_type: PlotType, name: str):
        self._client: Client = client
        self._plot_type: PlotType = plot_type
        self._name: str = name

    @property
    def plot_type(self) -> PlotType:
        return self._plot_type

    def _add_to_server(self) -> None:
        self._client.add_plot(self._plot_type, self._name)

    @abstractmethod
    def update(self, *args) -> None:
        pass


class TrainValLossTracker(Tracker):
    def __init__(self, client: Client, name: str):
        super(TrainValLossTracker, self).__init__(client, plot_type=PlotType.train_val_loss, name=name)
        self._train: List[float] = []
        self._val: List[float] = []
        self._epochs: List[int] = []

        self._add_to_server()

    def update(self, train: float, val: float, epoch: int) -> None:
        new_data: NDArray = np.array([train, val, epoch], dtype=np.float32)
        self._train.append(train)
        self._val.append(val)
        self._epochs.append(epoch)
        self._client.update_plot(self._name, new_data)
