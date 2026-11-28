import numpy as np
from src.operating_policy import validation_options


def test_constrained_options_meet_targets_and_support():
    y = np.array([0, 1, 0, 0, 1, 1, 0, 1])
    p = np.array([.1, .2, .3, .4, .5, .6, .7, .9])
    for row in validation_options(y, p, minimum_flags=2)[1:]:
        if row['constraint_met']:
            metric, _, _, target = row['rule'].split('_')
            assert row[metric] >= int(target)/100
            assert row['tp']+row['fp'] >= 2


def test_unattainable_constraint_does_not_invent_threshold():
    rows = validation_options([0, 0, 0, 1], [.5, .5, .5, .5], minimum_flags=2)
    precision50 = next(r for r in rows if r['rule']=='precision_at_least_50')
    assert not precision50['constraint_met']
    assert 'threshold' not in precision50
