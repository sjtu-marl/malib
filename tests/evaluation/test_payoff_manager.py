import pytest

from malib.utils.typing import RolloutFeedback
from malib.evaluator.utils.payoff_manager import PayoffManager


@pytest.mark.parametrize(
    "agents,solve_method",
    [
        # two players with FSP
        ([f"agent_{i}".format(i) for i in range(2)], "fictitious_play"),
        # two players with alpha-rank
        ([f"agent_{i}".format(i) for i in range(2)], "alpha_rank"),
        # multiplayers with alpha-rank
        ([f"agent_{i}".format(i) for i in range(3)], "alpha_rank"),
    ],
)
def test_payoff_manager(agents, solve_method):
    manager = PayoffManager(agent_names=agents, exp_cfg={}, solve_method=solve_method)

    # add new results
    policy_pool = {}
    policy_ids = [f"policy_{i}" for i in range(1)]
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
            tmp["agent_reward/{}_mean".format(aid)] = 0.1
        feedback = RolloutFeedback("test", None, tmp, comb)
        manager.update_payoff(feedback)

    # generate equilibrium
    manager.check_done(policy_pool)

    # aggregation
    eq = manager.compute_equilibrium(policy_pool)
    manager.update_equilibrium(policy_pool, eq)
    eq = manager.get_equilibrium(policy_pool)
    print(eq)
