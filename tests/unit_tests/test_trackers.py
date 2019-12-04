from unittest import TestCase
import numpy as np

from traintracker.trackers import TrainValLossTracker, AccuracyTracker


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


class TestAccuracyTracker(TestCase):
    def test_serverless_connection_update(self):
        at = AccuracyTracker("acc")

        # No exception raised = pass
        at.update(np.array([1, 0, 1]), np.array([0, 1, 1]), step=1)

    def test_accuracy_correct(self):
        at = AccuracyTracker("acc")
        labels = np.array([1, 1, 2, 2, 0])
        predictions = np.array([1, 1, 2, 0, 0])
        # acc should be 4/5 = .8
        true_acc = 4/5

        at.update(predictions, labels, 1)
        pred_acc = at.get_accuracies(as_np=True)[0]

        self.assertEqual(true_acc, pred_acc,
                         f"Accuracy Tracker's computed accuracy {pred_acc} != {true_acc}")
