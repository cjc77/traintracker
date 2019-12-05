from unittest import TestCase

from traintracker.server import Server
from traintracker.tracker_plots import SOURCE_FORMATS
from traintracker.util.defs import *


class TestServer(TestCase):
    def test_add_plot(self):
        name = "tvl"
        id_ = 12345678
        s = Server()
        s._add_plot(PlotType.train_val_loss, name, id_)
        p = s._plots[id_]
        src = SOURCE_FORMATS[PlotType.train_val_loss]
        self.assertTrue(p.source.data.keys() == src.keys(),
                        "Column source for new plot should have same keys as template.")
        self.assertTrue(all([not ls for ls in p.source.data.values()]),
                        "All values (lists) for column source for new plot should be empty.")
        self.assertFalse((not s._queues[id_]), "Queue for new plot's data should have been initialized")
