# This is stuff for "forcing the hand" of the iterations so I can see what worked well.

import enum
import math
import typing

import itertools


class Action(enum.IntEnum):
    remove = enum.auto()
    add = enum.auto()


class ChainComponent(typing.NamedTuple):
    action: Action
    start: float
    end: float

    def __str__(self):
        return f"{self.action.name.title()}({self.start} -> {self.end})"

    def __repr__(self):
        return f"ChainComponent(Action.{self.action.name}, {self.start}, {self.end})"

    def contains(self, action: Action, x: float):
        if action != self.action:
            return False

        return self.start <= x <= self.end


class Chains:
    # Try to make this a Monoid - hopefully I've done it right!
    _chains: typing.Sequence[ChainComponent]

    def _aggregate_components(
        self, chain_components: typing.List[ChainComponent]
    ) -> typing.List[ChainComponent]:

        temp_list = sorted(chain_components)

        def get_direction(chain_component: ChainComponent):
            return chain_component.action

        # Combine the add and remove stages separately.
        overall_merged = []
        for _, sub_iter in itertools.groupby(temp_list, get_direction):
            sub_list = list(sub_iter)

            merged = [sub_list[0]]

            for current in sub_list:
                head = merged.pop()
                if current.start <= head.end:
                    new_head = head._replace(end=max(head.end, current.end))
                    merged.append(new_head)

                else:
                    merged.append(head)
                    merged.append(current)

            overall_merged.extend(merged)

        return overall_merged

    def __init__(self, *chain_component: ChainComponent):
        self._chains = self._aggregate_components(chain_component)

    def __add__(self, other):
        # https://stackoverflow.com/a/43600953/11308690

        if not isinstance(other, Chains):
            raise TypeError(other)

        temp_list = list(self._chains) + list(other._chains)

        overall_merged = self._aggregate_components(temp_list)

        return Chains(*overall_merged)

    def __str__(self):
        bits = [str(x) for x in self._chains]
        if bits:
            return ", ".join(bits)

        else:
            return "<Empty Chain>"

    def __repr__(self):
        bits = [repr(x) for x in self._chains]
        comma_bits = ", ".join(bits)
        return f"Chains({comma_bits})"

    def contains(self, action: Action, x: float):
        return any(chain.contains(action, x) for chain in self._chains)


def make_chains(actions: typing.List[Action], n: int):

    one_span = 1.0 / n

    single_span_flags = [False, True]

    # Could be add, remove or both

    for span_flags in itertools.product(single_span_flags, repeat=n * len(actions)):
        all_chain_components = []

        for global_idx, one_flag in enumerate(span_flags):
            action_idx, idx = divmod(global_idx, n)
            action = actions[action_idx]
            start = idx * one_span
            end = (idx + 1) * one_span
            if math.isclose(end, 1.0):
                end = 1.0

            if one_flag:
                one_chain_component = ChainComponent(action, start, end)
                all_chain_components.append(one_chain_component)

        # Don't need to yield a do-nothing chain...
        if all_chain_components:
            yield Chains(*all_chain_components)


def make_single_sided_chains(n: int):
    """Make chains which only add or remove at a time"""

    for action in Action:
        yield from make_chains([action], n)


def make_all_chains(n: int):
    yield from make_chains(list(Action), n)


def test_chains():
    add1to5 = ChainComponent(Action.add, 1.0, 5.0)

    for should_combine, delta, action in (
        (True, -0.1, Action.add),
        (True, 0.0, Action.add),
        (False, 0.1, Action.add),
        (False, -0.1, Action.remove),
    ):
        add_next = ChainComponent(action, 5.0 + delta, 8.0)

        target_len = 1 if should_combine else 2
        # Try all in one
        chains = Chains(add1to5, add_next)
        assert len(chains._chains) == target_len

        # And try adding chains already combined.
        c1 = Chains(add1to5)
        c2 = Chains(add_next)
        c_comb = c1 + c2
        assert len(c_comb._chains) == target_len

        print(chains)
        print(c_comb)


if __name__ == "__main__":
    test_chains()

    for c in make_single_sided_chains(4):
        print(c)
