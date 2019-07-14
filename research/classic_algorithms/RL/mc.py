from itertools import dropwhile
import random


def get_trace(cid):
    cid = list(dropwhile(lambda dig: dig == 0, cid))
    states = list(map(lambda dig: (dig + 1) % 3, cid))
    rewards = list(map(lambda dig: dig % 2, cid))

    return states, rewards


def get_discounted(gamma, trace, from_idx):
    cnt = 0
    ret = 0
    for i in range(from_idx, len(trace[1])):
        ret = ret + (gamma ** cnt) * trace[1][i]
        cnt = cnt + 1
    return ret


def update_avg(new_val, old_avg, count):
    delta = float((new_val - old_avg)) / float(count + 1)
    return float(old_avg) + delta


def monte_carlo_first_visit(cid, states, iterations, gamma):
    vs = [random.uniform(-5, 5)] * states
    returns_avg = [0 for _ in range(states)]
    counts = [0 for _ in range(states)]

    for _ in range(iterations):
        ss, rs = get_trace(cid)
        for s in range(states):
            first_occ = next((i for i, x in enumerate(ss) if x == s), None)
            if first_occ is None:
                continue

            ret = get_discounted(gamma, (ss, rs), first_occ)
            returns_avg[s] = update_avg(ret, returns_avg[s], counts[s])
            counts[s] = counts[s] + 1
            vs[s] = returns_avg[s]

    return vs


if __name__ == '__main__':
    CID = [0, 1, 0, 7, 9, 9, 3, 1]
    print(monte_carlo_first_visit(CID, 3, 1, 1))
    # Output: [-3.333225212453599, 5.0, 6.0]
