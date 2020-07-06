def method_1(cmd):
    data = (1, 2)
    if cmd == 'add':
        return data[0] + data[1]
    elif cmd == 'mult':
        return data[0] * data[1]
    elif cmd == 'div':
        return data[0] / data[1]


def main():
    import time

    start_time = time.monotonic()
    print(time.ctime())
    for i in range(10000):
        method_1(cmd='add')
        method_1(cmd='add')
        method_1(cmd='add')
        method_1(cmd='mult')
        method_1(cmd='mult')
        method_1(cmd='div')

    print('minutes: ', (time.monotonic() - start_time) / 60)


if __name__ == "__main__":
    main()
