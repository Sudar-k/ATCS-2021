num = 4648

def add_digits(n):

    sum = 0
    while(sum > 10 or sum == 0):
        for l in str(n):
            sum += int(l)
        n = sum
        sum = 0
        if n<10:
            break
    print(n)

add_digits(num)