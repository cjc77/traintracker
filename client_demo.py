from train_tracker.client import Client, RandomTracker, TestLineTracker
from train_tracker.server import PORT
from train_tracker.util.defs import *
import time


def main():
    pc = Client()
    pc.connect("127.0.0.1", PORT)

    rt = RandomTracker(pc)
    tlt = TestLineTracker(pc)
    pc.start_plot_server()

    for i in range(10):
        rt.update()
        tlt.update(i, i * 2)
        time.sleep(1)

    pc.shutdown_server()
    pc.close_connection()


if __name__ == '__main__':
    main()
