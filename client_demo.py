from train_tracker.client import Client, RandomTracker, TrainValLossTracker
from train_tracker.server import PORT
from train_tracker.util.defs import *
import time


def main():
    pc = Client()
    pc.connect("127.0.0.1", PORT)

    lt = TrainValLossTracker(pc, "model1_tv_loss")
    pc.start_plot_server()

    for i in range(10):
        lt.update(10 - i * np.random.normal(), 10 - i * np.random.normal(), i)
        time.sleep(1)

    pc.shutdown_server()
    pc.close_connection()


if __name__ == '__main__':
    main()
