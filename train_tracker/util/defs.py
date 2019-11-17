from typing import Dict, Tuple, Optional, Union
from enum import IntEnum

PORT = 54321
PS_PORT = 12345
TIMEOUT = 100
BUFFSIZE = 1024
BYTEORDER = "little"
INT32 = 4


class PlotType(IntEnum):
    random = 100


class Cmd(IntEnum):
    server_shutdown = 1
    add_plot = 2
    start_plot_server = 3

