from itertools import dropwhile


def get_trace(cid):
    cid = list(dropwhile(lambda dig: dig == 0, cid))
    states = list(map(lambda dig: (dig + 1) % 3, cid))
    rewards = list(map(lambda dig: dig % 2, cid))
    trace = ' '.join(map(lambda pair: 's' + str(pair[0]) + ' '
                                      + str(pair[1]), zip(states, rewards)))
    return trace


if __name__ == '__main__':
    CID = [0, 1, 0, 7, 9, 9, 3, 1]
    print(get_trace(CID))
    # Output: s2 1 s1 0 s2 1 s1 1 s1 1 s1 1 s2 1
