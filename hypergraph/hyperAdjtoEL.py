# coding: utf-8
import sys

if __name__ == "__main__":
    edgelist = []
    try:
        _, fn, out = sys.argv[:3]
    except:
        print("usage: python hyperAdjtoEL.py [input file] [output file]")
        exit(1)
    with open(fn, "r") as f:
        f.readline()
        nv = int(f.readline().strip())
        mv = int(f.readline().strip())
        nh = int(f.readline().strip())
        mh = int(f.readline().strip())
        voff = []
        for i in range(nv):
            voff.append(int(f.readline()))
        voff.append(mv)
        vneigh = []
        for i in range(mv):
            vneigh.append(int(f.readline()))
        for i in range(nv):
            for neighi in range(voff[i], voff[i+1]):
                edgelist.append((i, vneigh[neighi] + nv))
        hoff = []
        for i in range(nh):
            hoff.append(int(f.readline()))
        hoff.append(mh)
        hneigh = []
        for i in range(mh):
            hneigh.append(int(f.readline()))
        for i in range(nh):
            for neighi in range(hoff[i], hoff[i+1]):
                edgelist.append((i+nv, hneigh[neighi]))
                
    with open(out, "w") as f:
        f.writelines("%i %i\n" %(u, v) for (u, v) in edgelist)
