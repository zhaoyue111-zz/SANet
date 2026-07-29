# -*- coding: utf-8 -*-

import json
import pandas as pd

def main(filepath):
    output_csv = []
    uuids = set()
    with open(filepath, "r") as finput:
        line = finput.readline()
        while line:
            l_header = line.find("|")
            if l_header >= 0:
                suuid = line[:l_header]
                label_contents = line[l_header+1:]
                print(label_contents)
                labels = json.loads(label_contents)

                for k, v in labels.iteritems():
                    print("*" * 60)
                    print("data {} have {} records".format(k.encode("utf-8"), len(v)))
                    for vv in v:
                        print("record {}".format(vv))

                        ssuuid = k.encode("utf-8")
                        this_data = [0] * 6
                        this_data[0] = ssuuid
                        uuids.add(ssuuid)

                        # this_data = [suuid]
                        for kkk, vvv in vv.iteritems():
                            if kkk == "x":
                                this_data[1] = float(vvv)
                            elif kkk == "y":
                                this_data[2] = float(vvv)
                            elif kkk == "z":
                                this_data[3] = float(vvv)
                            elif kkk == "radius":
                                this_data[4] = float(vvv) * 2
                            elif kkk == "status":
                                this_data[5] = vvv
                            else:
                                print("wrong key {}...".format(kkk))

                        output_csv.append(this_data)
            else:
                print("warning, line {} not formatted error".format(line))
            line = finput.readline()

    df = pd.DataFrame(output_csv)
    df.to_csv("./lung_data_batch_21.csv", header=["seriesuid","coordX","coordY","coordZ","diameter_mm", "nodule_type"], index=None)

    df = pd.DataFrame(list(uuids))
    df.to_csv("./seriesuids.csv", header=["seriesuid"], index=None)


if __name__ == "__main__":
    main("./lung_data_batch_21.txt")
