import numpy as np

with open('./Data/dict/TrainableMidi2-Table.txt', 'r') as f:
    buf = f.read()

forward_dict = eval(buf)
backward_dict = {}
for x,y in forward_dict.items():
    backward_dict[y] = x
    # print str(x)+' : '+str(y)

with open('./Data/a.txt', 'r') as f: # input src
    buf = f.read()

toks = buf.split()

with open('./Data/result/a.txt', 'w') as f: # output
    for tok in toks:
        f.write(backward_dict[int(tok)])