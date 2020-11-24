import os
import statistics
import pandas as pd
from datetime import datetime

def read_dvae_times(path):
    for fn in os.listdir(path):
        if not fn.endswith("txt"): continue
        with open(os.path.join(path, fn), "r") as f:
            ll = ""
            e = 0
            times = []
            for l in f:
                if l.startswith("Epoch:"):
                    i = int(l[7:l.index(",")])
                    if i > e:
                        assert i == e + 1
                        e += 1
                        if e > 1:
                            # extract time from prev
                            time_str = ll[ll.index("[")+1: ll.index("<")]
                            t = datetime.strptime(time_str, '%H:%M:%S')
                            times += [(t.hour * 60 + t.minute) * 60 + t.second]
                            # print('Time:', t.time(), (t.hour * 60 + t.minute) * 60 + t.second)
                    ll = l
            if times:
                t = sum(times) / len(times)
                print(fn)
                print(int(t))
                print(statistics.stdev(times))
                print("{:.2f}".format(t / 60))
                print("{:.2f}".format(t / 3600))
    pass


def read_ogbg_times(path):
    for fn in os.listdir(path):
        if not fn.endswith("csv"): continue
        times = pd.read_csv(os.path.join(path, fn), header=None)
        times = times[times.columns[2]].astype("float").tolist()
        t = sum(times) / len(times)
        print(fn)
        print(int(t))
        print(statistics.stdev(times))
        print("{:.2f}".format(t / 60))



if __name__ == "__main__":
    # read_dvae_times(".")

    read_ogbg_times("../ogbg-times")