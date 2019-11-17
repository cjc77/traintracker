from train_tracker.client import Client
from train_tracker.server import PORT
from train_tracker.util.defs import *
import time


def main():
    pc = Client("127.0.0.1", PORT)
    pc.connect()
    pc.add_plot(PlotType.random)
    pc.add_plot(PlotType.test_line_plt)
    pc.start_plot_server()
    pc.update_plot(PlotType.test_line_plt, (1, 2))
    time.sleep(5)
    pc.shutdown_server()
    pc.close_connection()


if __name__ == '__main__':
    main()
