"""Research entry points; statistical association is not a causal safety mechanism."""


def fit_safety_probe(activations, labels, groups):
    raise NotImplementedError('Fit probes on discovery task/layout groups; validate on held-out groups')


def select_causal_targets(probe_results, paired_rollouts):
    raise NotImplementedError('Run matched ablation/patching and random-target controls before selecting targets')
