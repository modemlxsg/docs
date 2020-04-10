import torch
import numpy as np
import tensorflow as tf

# colate_fn for dataloader
def collate_pad(batch):
    batch = np.array(batch)
    imgs = batch[:, 0]
    label_tensor = batch[:,1]
    label_encode = batch[:,2]

    max_img_width = max(imgs, key=lambda img: img.shape[1]).shape[1]
    max_label_width = len(max(label_encode, key=lambda label: len(label)))

    img_list = []
    label_encode_list = []
    for img, label, label_encode in batch:
        img = np.pad(img, ((0, 0), (0, max_img_width - img.shape[1]), (0, 0)), mode='constant')
        img_list.append(img)

        label_encode_list.extend(label_encode)

    img_tensor = torch.tensor(img_list, dtype=torch.float32).permute(0, 3, 1, 2)  # b,c,h,w
    label_encode = torch.tensor(label_encode_list, dtype=torch.int32)

    return img_tensor, label_tensor, label_encode

# # colate_fn for dataloader
# def collate_pad(batch):
#     batch = np.array(batch)
#     imgs = batch[:, 0]
#     label_encode = batch[:,2]

#     max_img_width = max(imgs, key=lambda img: img.shape[1]).shape[1]
#     max_label_width = len(max(label_encode, key=lambda label: len(label)))

#     img_list = []
#     label_encode_list = []
#     for img, label, label_encode in batch:
#         img = np.pad(img, ((0, 0), (0, max_img_width - img.shape[1]), (0, 0)), mode='constant')
#         img_list.append(img)
#         label_encode = np.pad(label_encode, ((0, max_label_width - label_encode.shape[0])), mode='constant')
#         label_encode_list.append(label_encode)

#     img_tensor = torch.tensor(img_list, dtype=torch.float32).permute(0, 3, 1, 2) # b,c,h,w
#     return img_tensor, batch[:, 1], torch.tensor(label_encode_list, dtype=torch.int32)
# ctc decode    
def decode(sequence):
    inputs = tf.constant(sequence)
    sequence_length = tf.constant([inputs.shape[0]] * inputs.shape[1])
    decoded, _ = tf.nn.ctc_greedy_decoder(inputs, sequence_length, merge_repeated=False)
    decoded = tf.sparse.to_dense(decoded[0]).numpy()
    return decoded

# 
def convert2str(seq, lexicon):
    results = []
    for ls in seq:
        string = ""
        for index in ls:
            if index == 0: continue
            string += lexicon[index]
        results.append(string)

    return results
  

