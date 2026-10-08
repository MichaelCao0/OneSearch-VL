"""Shared reward weights for launch preflight, rollout and final aggregation."""
from dataclasses import dataclass
import json
import math
import os


@dataclass(frozen=True)
class RewardWeights:
    accuracy: float = 0.6
    query: float = 0.2
    evgr: float = 0.2

    def __post_init__(self):
        values = (self.accuracy, self.query, self.evgr)
        if any(not math.isfinite(v) or not 0 <= v <= 1 for v in values):
            raise ValueError('Reward weights must be finite numbers within [0, 1]')
        if not math.isclose(sum(values), 1.0, rel_tol=0, abs_tol=1e-9):
            raise ValueError('Accuracy, Query and EVGR reward weights must sum to 1')

    @classmethod
    def from_env(cls, *, process_weight=None):
        defaults = cls()
        try:
            return cls(
                accuracy=float(os.getenv('REWARD_ACCURACY_WEIGHT', defaults.accuracy)),
                query=float(os.getenv('REWARD_QUERY_WEIGHT', defaults.query)),
                evgr=float(os.getenv('FORMAL_EVGR_PROCESS_WEIGHT', defaults.evgr)
                           if process_weight is None else process_weight),
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f'Invalid reward weight configuration: {exc}') from exc

    def combine(self, r_format, r_accuracy, r_query, r_evgr):
        values = (r_format, r_accuracy, r_query, r_evgr)
        if any(not math.isfinite(float(v)) or not 0 <= float(v) <= 1 for v in values):
            raise ValueError(f'Reward inputs must be finite numbers within [0, 1], got {values}')
        return float(r_format) * (
            self.accuracy * float(r_accuracy) + self.query * float(r_query)
            + self.evgr * float(r_evgr)
        )

    def base_reward(self, r_format, r_accuracy, r_query, *, query_enabled):
        """Provisional/no-EVGR reward: allocate inactive components to accuracy.

        The formal hook replaces this value using all three configured weights.
        This also preserves the existing validation behavior when EVGR is off.
        """
        return self.combine(r_format, r_accuracy,
                            r_query if query_enabled else r_accuracy, r_accuracy)

    def validate_enabled(self):
        enabled = os.getenv('QUERY_REWARD_ENABLED', '1').strip().lower() in {'1', 'true', 'yes'}
        if self.query > 0 and not enabled:
            raise ValueError('Positive REWARD_QUERY_WEIGHT requires QUERY_REWARD_ENABLED=1')


if __name__ == '__main__':
    weights = RewardWeights.from_env()
    weights.validate_enabled()
    print(json.dumps({'accuracy': weights.accuracy, 'query': weights.query,
                      'evgr': weights.evgr, 'trace': weights.evgr / 2,
                      'ground': weights.evgr / 2}))
