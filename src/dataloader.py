from torch.utils.data import Dataset
import torch

from scipy.io.wavfile import read
from tqdm import tqdm
import pickle
import random
import time
import os

MAX_AUDIO_VAL = 32768.0


def get_data_to_buffer(train_config):
    # check cache
    start = time.perf_counter()
    if os.path.isfile(train_config.cache_bufer):
        with open(train_config.cache_bufer, 'rb') as inp:
            buffer = pickle.load(inp)
            end = time.perf_counter()
        print('load buffer from cache')
        print("cost {:.2f}s to load all data into buffer.".format(end - start))
        return buffer

    buffer = list()
    audio_names = sorted(os.listdir(train_config.data_path))
    for i in tqdm(range(len(audio_names))):
        sampling_rate, audio = read(train_config.data_path + '/' + audio_names[i])
        audio = audio / MAX_AUDIO_VAL
        audio = torch.FloatTensor(audio)
        buffer.append(audio)
    end = time.perf_counter()
    print("cost {:.2f}s to load all data into buffer.".format(end - start))

    os.makedirs(os.path.dirname(train_config.cache_bufer), exist_ok=True)
    with open(train_config.cache_bufer, 'wb') as outp:
        pickle.dump(buffer, outp, pickle.HIGHEST_PROTOCOL)

    return buffer


class BufferDataset(Dataset):
    def __init__(self, buffer, length_wav):
        self.buffer = buffer
        self.length_wav = length_wav
        self.length_dataset = len(self.buffer)

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, idx):
        audio = self.buffer[idx].unsqueeze(0)

        if audio.size(1) >= self.length_wav:
            audio_start = random.randint(0, audio.size(1) - self.length_wav)
            audio = audio[:, audio_start:audio_start + self.length_wav]
        else:
            audio = torch.nn.functional.pad(audio, (0, self.length_wav - audio.size(1)), 'constant')

        return {"audio": audio.squeeze(0)}
