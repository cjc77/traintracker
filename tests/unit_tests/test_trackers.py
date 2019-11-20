from unittest import TestCase
import numpy as np

from traintracker.trackers import TrainValLossTracker


class TestTrainValLossTracker(TestCase):
    def test_serverless_connection_update(self):
        tvlt = TrainValLossTracker("tv_loss")

        # No exception raised = pass
        tvlt.update(1, 2, 3)

    def test_update_yields_accurate_items(self):
        tvlt = TrainValLossTracker("tv_loss")
        # train_lss = np.array([1, 2, 3, 4, 5])
        step = np.arange(1, 10)
        train_lss = step * 2
        val_lss = step * 3
        for i in range(len(train_lss)):
            tvlt.update(step[i], train_lss[i], val_lss[i])
        s, t, v = tvlt.get_all_tracked(as_np=True)
        msg = "{}\nshould match\n{} for {}"
        self.assertTrue(np.all(s == step), msg.format(step, s, "step"))
        self.assertTrue(np.all(t == train_lss), msg.format(train_lss, t, "train loss"))
        self.assertTrue(np.all(v == val_lss), msg.format(val_lss, v, "val loss"))
