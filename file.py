from random import randint

def g(lower, upper, n):
    return([[randint(lower, upper), randint(lower, upper)] for i in range(n)])

dataset = g(1, 250, 250) + g(500, 600, 250) + g(1000, 1200, 250) + g(1600, 1800, 250)
