from dataclasses import dataclass
import torch


@dataclass
class V1_config:
    upsample_rates = [8, 8, 2, 2]
    upsample_kernel_sizes = [16, 16, 4, 4]
    upsample_initial_channel = 512
    resblock_kernel_sizes = [3, 7, 11]
    resblock_dilation_sizes = [[1, 3, 5], [1, 3, 5], [1, 3, 5]]


@dataclass
class TrainConfig:
    checkpoint_path = "checkpoints"
    test_path = 'test_wav'
    data_path = 'data/LJSpeech-1.1/wavs'
    cache_bufer = 'cache/buffer.pkl'

    wandb_project = 'HiFi-GAN'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    batch_size = 16
    epochs = 20000

    learning_rate = 0.0002
    lr_decay = 0.999
    beta1 = 0.8
    beta2 = 0.99

    save_step = 100
    length_wav = 8192


model_config = V1_config()
train_config = TrainConfig()
