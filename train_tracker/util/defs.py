from typing import Dict, Tuple, List, Sequence, Optional, Union
from enum import IntEnum
import numpy as np

NDArray = np.ndarray

PORT = 54321
PS_PORT = 12345
TIMEOUT = 100
BUFFSIZE = 1024
BYTEORDER = "little"
INT32 = 4

NP_ORDER: Dict[str, str] = {
    "little": 'L',
    "big": 'B'
}


class PlotType(IntEnum):
    test_line_plt = 99
    random = 100


class Cmd(IntEnum):
    server_shutdown = 1
    add_plot = 2
    start_plot_server = 3
    update_plot = 4

