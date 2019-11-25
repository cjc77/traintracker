from traintracker.client import Client
from traintracker.trackers import TrainValLossTracker
from traintracker.server import PORT
from traintracker.util.defs import *
import time


def main():
    pc = Client()
    pc.connect("127.0.0.1", PORT)

    # If may want to run without server
    # init trackers
    # lt1 = TrainValLossTracker("model1_tv_loss")
    # lt2 = TrainValLossTracker("model2_tv_loss")
    l_ts = [TrainValLossTracker(f"model{i}_loss") for i in range(10)]
    # then connect
    # lt1.connect_client(pc)
    # lt2.connect_client(pc)
    for lt in l_ts:
        lt.connect_client(pc)

    # # OR if want to run with server for sure
    # # init trackers w/ client
    # lt1 = TrainValLossTracker("model1_tv_loss", pc)
    # lt2 = TrainValLossTracker("model2_tv_loss", pc)

    pc.start_plot_server()

    for i in range(100):
        # lt1.update(10 - i * np.random.normal(), 10 - i * np.random.normal(), i)
        # lt2.update(10 - i * np.random.normal(), 10 - i * np.random.normal(), i)
        for lt in l_ts:
            lt.update(10 - i * np.random.normal(), 10 - i * np.random.normal(), i)
        time.sleep(.2)
        # print("lt1:", lt1.get_steps(True), lt1.get_train_losses(True), lt1.get_val_losses(True), sep='\n')
        # print("lt2:", lt2.get_steps(True), lt2.get_train_losses(True), lt2.get_val_losses(True), sep='\n')

    pc.shutdown_server()
    pc.close_connection()


if __name__ == '__main__':
    main()
