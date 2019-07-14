def get_personalised_params(cid):
    x = cid[-3]
    y = cid[-2]
    z = cid[-1]

    j = (z + 1) % 3 + 1
    p = 0.25 + 0.5 * (x / 10.0)
    gamma = 0.3 + 0.5 * (y / 10.0)
    return j, p, gamma


if __name__ == '__main__':
    CID = [0, 1, 0, 7, 9, 9, 3, 1]
    print(get_personalised_params(CID))
    # Output: (3, 0.7, 0.44999999999999996)
