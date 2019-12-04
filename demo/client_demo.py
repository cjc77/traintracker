import time

from traintracker.client import Client
from traintracker.trackers import TrainValLossTracker, AccuracyTracker
from traintracker.server import PORT
from traintracker.util.defs import *


def main():
    pc = Client()
    pc.connect("127.0.0.1", PORT)

    # If may want to run without server
    # init trackers
    lt1 = TrainValLossTracker("model1_tv_loss")
    at1 = AccuracyTracker("model1_acc")
    # then connect
    lt1.connect_client(pc)
    at1.connect_client(pc)

    print(lt1.id, at1.id)

    pc.start_plot_server()

    for i in range(100):
        lt1.update(10 - i * np.random.normal(), 10 - i * np.random.normal(), i)
        at1.update(np.random.choice([0, 1], size=10, replace=True), 
                   np.random.choice([0, 1], size=10, replace=True),
                   i)
        time.sleep(.2)
        # print("lt1:", lt1.get_steps(True), lt1.get_train_losses(True), lt1.get_val_losses(True), sep='\n')

    pc.shutdown_server()
    pc.close_connection()


if __name__ == '__main__':
    main()
