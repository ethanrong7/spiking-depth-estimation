
import numpy as np
import cv2
import warnings



def read_depth(file_path):
    depth = cv2.imread(file_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
    depth = depth[:,:,0]
    
    assert np.sum(depth) > 0, 'Fail to load depth.'
    return depth


def depth2disp(depth, f, B):
    # Avoid division by zero by setting a minimum depth value
    min_depth = 1e-6
    depth = np.maximum(depth, min_depth)
    
    disp = (f * B) / depth
    return disp


def get_block_spikes(filename, begin_idx=0, spike_height=250, spike_width=400, block_len=50, flipud=False, with_head=False):

    file_reader = open(filename, 'rb')
    video_seq = file_reader.read()
    video_seq = np.frombuffer(video_seq, 'b')
    video_seq = np.array(video_seq).astype(np.uint8)
    img_size = spike_height * spike_width
    img_num = len(video_seq) // (img_size // 8)

    end_idx = begin_idx + block_len
    if end_idx > img_num:
        warnings.warn("block_len exceeding upper limit! Zeros will be padded in the end. ", ResourceWarning)
        end_idx = img_num

    SpikeMatrix = np.zeros([block_len, spike_height, spike_width], np.uint8)

    pix_id = np.arange(0, block_len * spike_height * spike_width)
    pix_id = np.reshape(pix_id, (block_len, spike_height, spike_width))
    comparator = np.left_shift(1, np.mod(pix_id, 8))
    byte_id = pix_id // 8
    id_start = begin_idx * img_size // 8
    id_end = id_start + block_len * img_size // 8
    data = video_seq[id_start:id_end]
    data_frame = data[byte_id]
    result = np.bitwise_and(data_frame, comparator)
    tmp_matrix = (result == comparator)

    if flipud:
        SpikeMatrix = tmp_matrix[:, ::-1, :]
    else:
        SpikeMatrix = tmp_matrix

    file_reader.close()
    return SpikeMatrix


def delete_time_head(video_seq):
    chunk_size_16 = 16
    chunk_size_13000 = 13000

    total_chunks = len(video_seq) // (chunk_size_16 + chunk_size_13000)

    new_video_seq = []

    offset = 0
    for _ in range(total_chunks):
        offset += chunk_size_16
        new_video_seq.append(video_seq[offset:offset + chunk_size_13000])
        offset += chunk_size_13000

    video_seq = np.concatenate(new_video_seq)
    return video_seq


def get_spike_matrix(filename, spike_height=250, spike_width=416, flipud=True, with_head=False, Time_head=False):

    file_reader = open(filename, 'rb')
    video_seq = file_reader.read()
    video_seq = np.frombuffer(video_seq, 'b')

    video_seq = np.array(video_seq).astype(np.byte)
    
    if Time_head:
        video_seq = delete_time_head(video_seq)

    img_size = spike_height * spike_width
    img_num = len(video_seq) // (img_size // 8)

    SpikeMatrix = np.zeros([img_num, spike_height, spike_width], np.byte)

    pix_id = np.arange(0, spike_height * spike_width)
    pix_id = np.reshape(pix_id, (spike_height, spike_width))
    comparator = np.left_shift(1, np.mod(pix_id, 8))
    byte_id = pix_id // 8

    for img_id in np.arange(img_num):
        id_start = img_id * img_size // 8
        id_end = id_start + img_size // 8
        cur_info = video_seq[id_start:id_end]
        data = cur_info[byte_id]
        result = np.bitwise_and(data, comparator)
        if flipud:
            SpikeMatrix[img_id, :, :] = np.flipud((result == comparator))
        else:
            SpikeMatrix[img_id, :, :] = (result == comparator)
    file_reader.close()

    return SpikeMatrix