import time

from traintracker.client import Client
from traintracker.trackers import TrainValLossTracker, AccuracyTracker, ConfusionMatrixTracker
from traintracker.server import PORT
from traintracker.util.defs import *


def main():
    # for the confusion matrix
    levels = [0, 1, 2]
    m = 100

    pc = Client()
    pc.connect("127.0.0.1", PORT)

    # If may want to run without server
    # init trackers
    lt1 = TrainValLossTracker("model1_tv_loss")
    at1 = AccuracyTracker("model1_acc")
    cm1 = ConfusionMatrixTracker(m, levels, "cm1")
    # then connect
    lt1.connect_client(pc)
    at1.connect_client(pc)
    cm1.connect_client(pc)

    pc.start_plot_server()

    for i in range(100):
        lt1.update(10 - i * np.random.normal(), 10 - i * np.random.normal(), i)
        at1.update(np.random.choice([0, 1], size=10, replace=True), 
                   np.random.choice([0, 1], size=10, replace=True),
                   i)
        pred = np.random.choice(levels, size=m)
        true = np.random.choice(levels, size=m)
        cm1.update(pred, true)

        time.sleep(.2)

    pc.shutdown_server()
    pc.close_connection()


if __name__ == '__main__':
    main()
