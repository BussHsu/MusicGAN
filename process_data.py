import numpy as np

DEFAULT_EOF_SYM = 100
SEGMENT_LEN = 20
SEGMENT_SHIFT = 12

# cut music into segments
def segment_music(music, seg_len = SEGMENT_LEN, seg_shift = SEGMENT_SHIFT):
    assert len(music)>=seg_len , 'Error: segment_music, input music length too short'
    seg_list = []
    music_len = len(music)
    overhead = seg_len - seg_shift
    n_segs = (music_len-overhead) / seg_shift

    for cnt in xrange(n_segs):
        seg_list.append(music[seg_shift*cnt:seg_shift*(cnt+1)+overhead])

    if (music_len-overhead)%seg_shift!=0:
        seg_list.append(music[-seg_len:])

    return seg_list



# determine data range
data_arry = np.genfromtxt('./Data/raw/TrainableMidi2.txt',delimiter=' ')
# data_min = data_arry.min()
# eof_sym = data_min-1
eof_sym = 96
data_arry[np.where(data_arry == DEFAULT_EOF_SYM)] = eof_sym
#
#
# # rearrange with eof symbol set to 0
# shift_amount = 0 - eof_sym
# data_arry += shift_amount
data_max = data_arry.max()
print data_max
# data_arry = data_arry[:,np.newaxis]
# np.savetxt('./Data/temp/shifted.dat', data_arry.T, delimiter=' ', fmt='%d')

# # create raw segment by splitting '0'
# seg_strt = 0
# music_list = []
# for curr_idx in xrange(len(data_arry)):
#     if data_arry[curr_idx] == 0:
#         music_list.append(data_arry[seg_strt:curr_idx + 1])
#         seg_strt = curr_idx+1
#
#
# # remove too short music
# music_list = [x for x in music_list if len(x) >= SEGMENT_LEN]
#
# seg_list = []
# for music in music_list:
#     seg_list.extend(segment_music(music,SEGMENT_LEN,SEGMENT_SHIFT))
#
# data_mat = np.empty((len(seg_list), SEGMENT_LEN),dtype=np.uint8)
#
# for idx, segment in enumerate(seg_list):
#     data_mat[idx,:] = segment

np.savetxt('./Data/midi2.dat', data_arry, delimiter=' ', fmt='%d')