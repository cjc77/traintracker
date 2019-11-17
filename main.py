from train_tracker.train_tracker import TrainTracker, PlotType


def main():
    ps = TrainTracker()
    ps.add_plot(PlotType.random)
    ps.start()


if __name__ == '__main__':
    main()
