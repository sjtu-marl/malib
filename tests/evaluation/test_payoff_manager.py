import pytest

from malib.utils.typing import RolloutFeedback
from malib.evaluator.utils.payoff_manager import PayoffManager


@pytest.mark.parametrize(
    "agents,solve_method",
    [
        # two players with FSP
        ([f"agent_{i}".format(i) for i in range(2)], "fictitious_play"),
        # two players with alpha-rank
        ([f"agent_{i}".format(i) for i in range(2)], "alpharank"),
        # multiplayers with alpha-rank
        ([f"agent_{i}".format(i) for i in range(3)], "alpharank"),
    ],
)
@pytest.mark.parametrize("policy_num", [1, 3])
def test_payoff_manager(agents, solve_method, policy_num):
    manager = PayoffManager(agent_names=agents, exp_cfg={}, solve_method=solve_method)

    # add new results
    policy_pool = {}
    policy_ids = [f"policy_{i}" for i in range(policy_num)]
    matchups = []
    for agent in agents:
        policy_pool[agent] = policy_ids
        matchups.extend(
            manager.get_pending_matchups(
                agent_name=agent, policy_id="policy_0", policy_config={}
            )
        )

    feedbacks = []
    for comb in matchups:
        tmp = {}
        for aid, (pid, _) in comb.items():
            tmp["reward/{}_mean".format(aid)] = 0.1
        feedback = RolloutFeedback(
            "test", None, tmp, {k: pid for k, (pid, _) in comb.items()}
        )
        manager.update_payoff(feedback)

    # generate equilibrium
    manager.check_done(policy_pool)

    # aggregation
    eq = manager.compute_equilibrium(policy_pool)
    manager.update_equilibrium(policy_pool, eq)
    eq = manager.get_equilibrium(policy_pool)

    # try to do aggregation
    for brs in [None, dict.fromkeys(agents, f"policy_{policy_num}")]:
        manager.aggregate(eq, brs)
