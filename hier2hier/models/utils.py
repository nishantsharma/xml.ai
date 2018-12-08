
def invertPermutation(perm):
    inverse = [0] * len(perm)
    for i, p in enumerate(perm):
        inverse[p] = i
    return inverse

def onehotencode(n, i):
    return [1 if j==i else 0 for j in range(n)]
