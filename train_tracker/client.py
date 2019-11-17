import socket

from train_tracker.util.defs import *


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

    def send_cmd(self, cmd: Cmd) -> None:
        if self._socket:
            self._socket.sendall(cmd.to_bytes(INT32, BYTEORDER))
            ack = self._socket.recv(BUFFSIZE)
            ack = int.from_bytes(ack, BYTEORDER)
            assert ack == cmd, "Correct command not received by server."
