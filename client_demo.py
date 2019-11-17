from train_tracker.client import Client
from train_tracker.server import PORT
from train_tracker.util.defs import *
import time


def main():
    pc = Client("127.0.0.1", PORT)
    pc.connect()
    pc.add_plot(PlotType.random)
    pc.start_plot_server()
    time.sleep(10)
    pc.shutdown_server()
    pc.close_connection()


if __name__ == '__main__':
    main()
