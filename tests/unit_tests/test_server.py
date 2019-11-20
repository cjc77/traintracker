from unittest import TestCase

from traintracker.server import Server
from traintracker.util.defs import *


class TestServer(TestCase):
    def test_add_plot(self):
        name = "tvl"
        s = Server()
        s._add_plot(PlotType.train_val_loss, name)
        p = s._plots[name]
        src = s._source_formats[PlotType.train_val_loss]
        self.assertTrue(p.source.data.keys() == src.keys(),
                        "Column source for new plot should have same keys as template.")
        self.assertTrue(all([not ls for ls in p.source.data.values()]),
                        "All values (lists) for column source for new plot should be empty.")
        self.assertFalse((not s._queues[name]), "Queue for new plot's data should have been initialized")
