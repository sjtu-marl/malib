.. _api-doc:
.. automodule:: malib.algorithm

API Documentation
=================

Training APIs
-------------

We implemented five basic traning interfaces as below, users can find more details from :ref:`malib.agent`.


Algorithm Customization
-----------------------

MALib decouple the algorithm :ref:`malib.algorithm`


.. _api-environment-custom:

Environment Customization
-------------------------

MALib works with environments implemented based on `PettingZoo <http://pettingzoo.ml/>`_ interfaces.


Rollout Manners
^^^^^^^^^^^^^^^

There are two rollout manners in MALib, i.e., sequential and simultaneous.

Vectorized
^^^^^^^^^^

MALib will auto-vectorized for environments which support simultaneous rollout to do batch rollout and evaluation. You can modify the batch level by specifing the ``num_episodes`` in the :ref:`global-settings`.