nums = [3, 1, 2, 10, 1]


def running_sum(numbers):
    new = [numbers[0]]
    for i in range(1, len(numbers)):
        new.append(numbers[i] + new[i - 1])
    return new


print(running_sum(nums))
