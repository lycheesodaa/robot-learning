"""Paired scene construction requires a concrete simulator implementation."""


def generate_pairs(environment, task, initial_state, factor, seed):
    raise NotImplementedError('Change one named safety factor while controlling task and initial state; '
                              'record pair/layout IDs and label contacts/constraints from simulator state')
