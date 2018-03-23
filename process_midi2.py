import numpy as np

DEFAULT_EOF_SYM = 100
SEGMENT_LEN = 30
SEGMENT_SHIFT = 20

# cut music into segments
def segment_music(music, eof_sym, seg_len = SEGMENT_LEN, seg_shift = SEGMENT_SHIFT):
    assert len(music)>=seg_len , 'Error: segment_music, input music length too short'
    seg_list = []
    music_len = len(music)
    overhead = seg_len - seg_shift
    n_segs = (music_len-overhead) / seg_shift

    for cnt in xrange(n_segs):
        seg_list.append(music[seg_shift*cnt:seg_shift*(cnt+1)+overhead])

    left_over = (music_len-overhead)%seg_shift
    if left_over!=0:
        num_pad = seg_shift-left_over
        tmp = np.hstack((music[seg_shift*(cnt+1):],eof_sym*np.ones((num_pad),dtype=np.uint8)))
        assert len(tmp)==SEGMENT_LEN
        seg_list.append(tmp)

    return seg_list



# determine data range
tr_output_fpath = './Data/all_reel.dat'
te_output_fpath = './Data/all_reel_eval.dat'
data_arry = np.genfromtxt('./Data/raw/AllReel.txt',delimiter=' ')
eof_sym = 96
data_arry[np.where(data_arry == DEFAULT_EOF_SYM)] = eof_sym

data_arry = data_arry.flatten()
# data_arry = data_arry[:,np.newaxis]
# np.savetxt('./Data/temp/shifted.dat', data_arry.T, delimiter=' ', fmt='%d')

# # create raw segment by splitting eof_sym
seg_strt = 0
music_list = []
for curr_idx in xrange(len(data_arry)):
    if data_arry[curr_idx] == eof_sym:
        music_list.append(data_arry[seg_strt:curr_idx + 1])
        seg_strt = curr_idx+1


# remove too short music
music_list = [x for x in music_list if len(x) >= SEGMENT_LEN]

seg_list = []
for music in music_list:
    seg_list.extend(segment_music(music,eof_sym,SEGMENT_LEN,SEGMENT_SHIFT))

data_mat = np.empty((len(seg_list), SEGMENT_LEN),dtype=np.uint8)

for idx, segment in enumerate(seg_list):
    data_mat[idx,:] = segment


total_num = len(seg_list)
tr_num = int(0.8*total_num)

np.random.shuffle(data_mat)
tr_mat = data_mat[:tr_num,:]
te_mat = data_mat[tr_num:,:]

np.savetxt(tr_output_fpath, tr_mat, delimiter=' ', fmt='%d')
np.savetxt(te_output_fpath, tr_mat, delimiter=' ', fmt='%d')