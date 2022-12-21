import numpy as np
import torch

from scipy.io.wavfile import read
import torchaudio
import argparse
import os
import sys
sys.path.append('src/')

from model import Generator
from config import model_config, train_config
from utils import MelSpectrogram, MelSpectrogramConfig


def get_load_func(mel_spec, sr, device):
    def load_file(path):
        other_path, file_extension = os.path.splitext(path)
        other_path, filename = os.path.split(other_path)
        if file_extension == '.wav':
            sampling_rate, audio = read(path)
            audio = torch.FloatTensor(audio)
            return mel_spec(audio.unsqueeze(0).to(device)).cpu(), sampling_rate, filename
        if file_extension == '.npy':
            audio = np.load(path)
            return torch.from_numpy(audio).to(device), sr, filename
        raise Exception("Error type file")
    return load_file


def main(args):
    model = Generator(model_config)
    model = model.to(train_config.device)
    mel_spec = MelSpectrogram(MelSpectrogramConfig).to(train_config.device)

    checkpoint = torch.load(args.checkpoint, map_location=train_config.device)
    model.load_state_dict(checkpoint['generator'])
    model.eval()
    model.remove_weight_norm()
    
    load_file = get_load_func(mel_spec, args.sampling_rate, train_config.device)
    audio = []
    if args.file is not None:
        audio.append(load_file(args.file))
    if args.path is not None:
        for file in os.listdir(args.path):
            audio.append(load_file(os.path.join(args.path, file)))
    
    assert len(audio) > 0, 'Need at least one audio'
    os.makedirs(args.output, exist_ok=True)
    
    for i in range(len(audio)):
        mel, sr, name = audio[i]
        if len(mel.shape) == 2:
            mel = mel.T.unsqueeze(0)
        mel = mel.to(train_config.device)
        new_wav = model(mel)
        
        os.makedirs(args.output, exist_ok=True)
        filename = f"{name}_{i}.wav"
        torchaudio.save(os.path.join(args.output, filename), new_wav.squeeze(1).cpu(), sr)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description="TTS Project")
    args.add_argument(
        "-c",
        "--checkpoint",
        default='checkpoints/base_model.pth',
        type=str,
        help="the path to the checkpoint of the model",
    )
    args.add_argument(
        "-p",
        "--path",
        default='inference_data',
        type=str,
        help="the path to the directory with audio or spectograms",
    )
    args.add_argument(
        "-f",
        "--file",
        default=None,
        type=str,
        help="the path to the file for inference",
    )
    args.add_argument(
        "-sr",
        "--sampling_rate",
        default=22050,
        type=int,
        help="default sampling rate for spectograms",
    )
    args.add_argument(
        "-o",
        "--output",
        default="results",
        type=str,
        help="the path to save synthesized speech",
    )
    args = args.parse_args()
    main(args)
