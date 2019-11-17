from train_tracker.client import Client
from train_tracker.server import PORT
from train_tracker.util.defs import *


def main():
    pc = Client("127.0.0.1", PORT)
    pc.connect()
    pc.send_cmd(Cmd.add_plot)
    pc.send_cmd(Cmd.add_plot)
    pc.send_cmd(Cmd.add_plot)
    pc.send_cmd(Cmd.server_shutdown)
    pc.close_connection()


if __name__ == '__main__':
    main()
