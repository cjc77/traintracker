API
===

Server
------

.. currentmodule:: traintracker.server

.. autosummary::
   Server

.. autosummary::
   Server.run

.. autoclass:: Server
   :members:
   :private-members:
   :inherited-members:

Client
------

.. currentmodule:: traintracker.client

.. autosummary::
   Client

.. autosummary::
   Client.connect
   Client.close_connection
   Client.start_plot_server
   Client.shutdown_server

.. autoclass:: Client
   :members:
   :private-members:
   :inherited-members:

Trackers
--------

.. py:currentmodule:: traintracker.trackers

.. autosummary::
   Tracker.update
   Tracker.get_all_tracked

.. autoclass:: Tracker
   :members:
   :private-members:
   :inherited-members:


TrainValLossTracker
^^^^^^^^^^^^^^^^^^^

.. autosummary::
   TrainValLossTracker

.. autosummary::
   TrainValLossTracker.get_train_losses
   TrainValLossTracker.get_val_losses
   TrainValLossTracker.get_steps
   TrainValLossTracker.get_all_tracked
   TrainValLossTracker.update

.. autoclass:: TrainValLossTracker
   :members:
   :private-members:
   :inherited-members:


AccuracyTracker
^^^^^^^^^^^^^^^

.. autosummary::
   AccuracyTracker

.. autosummary::
   AccuracyTracker.get_accuracies
   AccuracyTracker.get_steps
   AccuracyTracker.get_all_tracked
   AccuracyTracker.update

.. autoclass:: AccuracyTracker
   :members:
   :private-members:
   :inherited-members:

