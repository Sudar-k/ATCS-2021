names = ['Eli', 'Sudar', 'Charlie', 'John']


def crowd_test(n):
    if len(n) > 3:
        print('The room is crowded')
    else:
        print("The room isn't crowded")

crowd_test(names)

names.pop(2)
names.pop(2)

crowd_test(names)