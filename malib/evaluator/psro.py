"""
Implementation of global evaluator for Policy Space Response Oracle (PSRO) algorithms. This evaluator will evaluate the
exploitablility between weighted payoff and an oracle payoff.
"""


from malib.evaluator.base_evaluator import BaseEvaluator
from malib.utils.formatter import pretty_print
from malib.utils.typing import RolloutFeedback, EvaluateResult, TrainingFeedback, Union


class PSROEvaluator(BaseEvaluator):
    """Evaluator for Policy Space Response Oracle algorithms"""

    class StopMetrics:
        """Supported stopping metrics"""

        MAX_ITERATION = "max_iteration"
        """Max iteration"""

        PAYOFF_DIFF_THRESHOLD = "payoff_diff_threshold"
        """Threshold of difference between the estimated payoff of best response and NE's"""

        NASH_COV = "nash cov"

    def __init__(self, **config):
        """Create a PSRO evaluator instance.

        :param Dict[str,Any] config: A dictionary of stopping metrics.
        """

        stop_metrics = config.get("stop_metrics", {})
        super(PSROEvaluator, self).__init__(stop_metrics, name="PSRO")

        self._iteration = 0

    def evaluate(
        self,
        content: Union[RolloutFeedback, TrainingFeedback],
        weighted_payoffs,
        oracle_payoffs,
        trainable_mapping=None,
    ):
        """Evaluate global convergence by comparing the margin between Nash and best response.
        Or, an estimation of exploitability
        """

        res = EvaluateResult.default_result()

        res[EvaluateResult.AVE_REWARD] = {
            aid: weighted_payoffs[aid] for aid in trainable_mapping
        }

        nash_cov = 0.0
        for aid, weighted_payoff in weighted_payoffs.items():
            nash_cov += abs(weighted_payoff - oracle_payoffs[aid])
        nash_cov /= 2.0

        res["exploitability"] = nash_cov

        # default by no limitation on iteration
        res[
            EvaluateResult.REACHED_MAX_ITERATION
        ] = self._iteration == self._metrics.get(
            PSROEvaluator.StopMetrics.MAX_ITERATION, 100
        )

        if res[EvaluateResult.REACHED_MAX_ITERATION]:
            res[EvaluateResult.CONVERGED] = True

        self._iteration += 1
        return res
