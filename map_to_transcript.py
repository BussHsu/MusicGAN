import numpy as np

with open('./Data/dict/TrainableMidi2-Table.txt', 'r') as f:
    buf = f.read()

forward_dict = eval(buf)
backward_dict = {}
for x,y in forward_dict.items():
    backward_dict[y] = x

with open('/media/yhsu/Jerry2/real_music.txt', 'r') as f:
    buf = f.read()

toks = buf.split()

with open('./Data/result/real_music.txt', 'w') as f:
    for tok in toks:
        f.write(backward_dict[int(tok)])