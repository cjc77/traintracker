from train_tracker.server import Server, PORT


def main():
    ps = Server("127.0.0.1", PORT)
    ps.run()


if __name__ == '__main__':
    main()
