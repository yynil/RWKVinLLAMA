data_file = '/media/yueyulin/data_4t/data/ultra_data/ultrachat/input_ids-002.pt'
label_file = '/media/yueyulin/data_4t/data/ultra_data/ultrachat/labels-001.pt'
import torch
input_ids = torch.load(data_file)
labels = torch.load(label_file)
print(input_ids.shape)
print(labels.shape)
print(input_ids[0].tolist())
print('----------------')
print(labels[0].tolist())