import sys
from sys import stdin

thread_to_col = {1: None, 2:8, 4:9, 8:10, 16:11, 32:12, 48:13, 99:14, 999:15}
NUM_COLS = 16
HYGRA = 1
GRAPHIT = 2

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: output-reader.py [filename]")
    _, fn = sys.argv
    lines = []
    tf = open(fn+'.csv', 'w+')
    rf = open(fn+'.raw', 'w+')
    r1 = "graph,V,H,mv,mh,parallel,,,,threads,,,,,,\n"
    r2 = ",,,,,framework,framework,schedule,1,2,4,8,16,32,48,numa\n"
    tf.write(r1)
    tf.write(r2)
    tf.flush()
    compiler = stdin.readline()
    rf.write(compiler)
    compiler = compiler.strip()
    gn = stdin.readline().strip()
    stdin.readline()
    v = stdin.readline().strip()
    mv = stdin.readline().strip()
    h = stdin.readline().strip()
    mh = stdin.readline().strip()
    rf.write("Graph %s has v=%s, h=%s, mv=%s, mh=%s\n" %(gn, v, h, mv, mh))
    rf.flush()
    rf.write(stdin.readline())
    framework = HYGRA
    time = []
    row = [""]*NUM_COLS
    row[:6] = gn, v, h, mv, mh, compiler
    rownum = 0
    for line in stdin:
        rf.write(line)
        rf.flush()
        line = line.strip()
        if framework == HYGRA:
            if line[:2] == "./":
                if rownum == 0:
                    row[6] = "hygra"
                else:
                    tf.write(",".join(row) + "\n")
                    tf.flush()
                    row = [""]*NUM_COLS
                rownum += 1
                row[7] = line
            elif line[:7] == "threads":
                threads = int(line.split("=")[-1])  # don't have to write lol
                col = thread_to_col[threads]
                if col is not None:
                    row[col] = "%.3g" % (sum(time)/len(time))
                time = []
            elif line[:7] == "graphit":
                tf.write(",".join(row) + "\n")
                tf.flush()
                row = [""] * NUM_COLS
                framework = GRAPHIT
                rownum = 0
            else:
                time.append(float(line.split(':')[-1]))
        elif framework == GRAPHIT:
            if line[-3:] == ".gt":
                if rownum == 0:
                    row[6] = "graphit"
                else:
                    tf.write(",".join(row) + "\n")
                    tf.flush()
                    row = [""]*NUM_COLS
                rownum += 1
                row[7] = line.split('/')[-1][:-3]
            elif line[:7] == "threads":
                threads = int(line.split("=")[-1])  # don't have to write lol
                col = thread_to_col[threads]
                if col is not None:
                    row[col] = "%.3g" % (sum(time[1:])/len(time[1:]))
                time = []
            else:
                time.append(float(line))
    tf.write(",".join(row) + "\n")
    tf.flush()
    tf.close()
    rf.close()
