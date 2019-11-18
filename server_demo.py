from train_tracker.server import Server, PORT, PS_PORT


def main():
    ps = Server()
    ps.run("127.0.0.1", port=PORT, plots_port=PS_PORT)


if __name__ == '__main__':
    main()
