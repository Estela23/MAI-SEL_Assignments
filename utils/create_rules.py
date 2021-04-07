def is_unique(s):
    a = s.to_numpy()
    return (a[0] == a).all(), a[0]


