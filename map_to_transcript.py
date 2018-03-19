import numpy as np

with open('./Data/dict/ReelJigTable.txt', 'r') as f:
    buf = f.read()

forward_dict = eval(buf)
backward_dict = {}
for x,y in forward_dict.items():
    backward_dict[y] = x

with open('./Data/generated/3.txt', 'r') as f:
    buf = f.read()

toks = buf.split()

with open('./Data/result/jig3.txt', 'w') as f:
    for tok in toks:
        f.write(backward_dict[int(tok)])