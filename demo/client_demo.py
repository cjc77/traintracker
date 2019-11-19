from train_tracker.client import Client, TrainValLossTracker
from train_tracker.server import PORT
from train_tracker.util.defs import *
import time


def main():
    pc = Client()
    pc.connect("127.0.0.1", PORT)

    # If may want to run without server
    # init trackers
    lt1 = TrainValLossTracker("model1_tv_loss")
    lt2 = TrainValLossTracker("model2_tv_loss")
    # then connect
    lt1.connect_client(pc)
    lt2.connect_client(pc)

    # # OR if want to run with server for sure
    # # init trackers w/ client
    # lt1 = TrainValLossTracker("model1_tv_loss", pc)
    # lt2 = TrainValLossTracker("model2_tv_loss", pc)

    pc.start_plot_server()

    for i in range(10):
        lt1.update(10 - i * np.random.normal(), 10 - i * np.random.normal(), i)
        lt2.update(10 - i * np.random.normal(), 10 - i * np.random.normal(), i)
        time.sleep(1)
        # print("lt1:", lt1.epochs, lt1.train_losses, lt1.val_losses, sep='\n')
        # print("lt2:", lt2.epochs, lt2.train_losses, lt2.val_losses, sep='\n')

    pc.shutdown_server()
    pc.close_connection()


if __name__ == '__main__':
    main()
