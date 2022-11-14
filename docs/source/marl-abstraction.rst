.. _marl-abstraction-doc:

.. role:: python(code)
  :language: python
  :class: highlight


MARL Training Paradigms
=======================

There are four kinds of training paradigms in the research of multi-agent reinforcement learning.

* **Independent Learning**: agents in an environment do not share policies, and no coordination between them, all of them optimize their own policies in an indenedent manner. Obviously, such a learning paradigm is also the learning paradigm for single-agent cases.

* **Centralized Training Decentralized Execution**: agents play their policies as single-agent cases, i.e., no coordination in the inference stage, but the training requires information from all of them, i.e., a shared critic or non-shared critics but share the information.

* **Fully Centrlized Learning**: agents are capsulated into a single policy/network, the training and inference behavior as a big **team agent**.

* **Networked (distributed) Learning**: this training paradigms is designed for some cases that involves hundreds of agents, in this case, the agent coordinate with only its neighbors (mathematically).

To cover the existing research in multi-agent reinforcement learning, we abstract the training paradigm as :python:`AgentInterface`. An AgentInterface is responsbile for the coordination between parameter server and dataset server, sometimes the other agent interfaces.

Independent AgentInterface
^^^^^^^^^^^^^^^^^^^^^^^^^^

See :ref:`api/malib.agent.indepdent\_agent`

Team AgentInterface
^^^^^^^^^^^^^^^^^^^

See :ref:`api/malib.agent.indepdent\_agent`

Asynchronous AgentInterface
^^^^^^^^^^^^^^^^^^^^^^^^^^^

See :ref:`api/malib.agent.indepdent\_agent`