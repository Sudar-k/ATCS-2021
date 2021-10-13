import itertools

nums, target = [3, 2, 4], 6


def two_sum(nums, target):
    pairs = list(itertools.combinations(nums, 2))
    print(pairs)
    for pair in pairs:
        if sum(pair) == target:
            ans = [nums.index(pair[0]), nums.index(pair[1])]
        continue
    print(ans)


two_sum(nums, target)
