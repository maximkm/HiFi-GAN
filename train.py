from torch.utils.data import DataLoader
from torch.nn import functional as F
import torchaudio
import torch

from scipy.io.wavfile import read
from tqdm import tqdm
import itertools
import argparse
import os
import sys

sys.path.append('src/')

from config import train_config, model_config
from dataloader import get_data_to_buffer, BufferDataset
from wandb_writer import WanDBWriter
from model import Generator, MultiPeriodDiscriminator, MultiScaleDiscriminator
from losses import feature_loss, generator_loss, discriminator_loss
from utils import MelSpectrogram, MelSpectrogramConfig


MEL_PAD_VAL = MelSpectrogramConfig().pad_value

def validate():
    audios = []
    for filename in os.listdir(train_config.test_path):
        sampling_rate, audio = read(train_config.test_path + '/' + filename)
        audio = torch.FloatTensor(audio)
        audios.append((audio, sampling_rate, filename))

    @torch.inference_mode()
    def wandb_log_eval(filepath, logger, mel_spec, device):
        generator = Generator(model_config).to(train_config.device)
        generator.load_state_dict(torch.load(filepath, map_location=device)['generator'])
        generator.eval()
        generator.remove_weight_norm()

        loss_mels = 0.
        for audio, sr, filename in audios:
            mel = mel_spec(audio.unsqueeze(0).to(device))
            new_wav = generator(mel)
            g_mel = mel_spec(new_wav)

            os.makedirs("tmp_results", exist_ok=True)
            path = f"tmp_results/{filename}_hifi_gan.wav"
            torchaudio.save(path, new_wav.squeeze(1).cpu(), sr)

            mel = F.pad(mel, (0, g_mel.size(-1) - mel.size(-1)), value=MEL_PAD_VAL).unsqueeze(1)
            loss_mel = F.l1_loss(mel, g_mel)
            loss_mels += loss_mel.item()

            logger.add_audio(filename, new_wav.squeeze(1).cpu().numpy().T, sr)
            logger.add_audio(f'{filename}_target', audio.unsqueeze(0).numpy().T, sr)
        logger.add_scalar("val_mel_loss", loss_mels / len(audios))

    return wandb_log_eval


def main(args):
    buffer = get_data_to_buffer(train_config)
    dataset = BufferDataset(buffer, train_config.length_wav)
    training_loader = DataLoader(
        dataset,
        batch_size=train_config.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=4
    )

    generator = Generator(model_config).to(train_config.device)
    mpd = MultiPeriodDiscriminator().to(train_config.device)
    msd = MultiScaleDiscriminator().to(train_config.device)
    mel_spec = MelSpectrogram(MelSpectrogramConfig).to(train_config.device)

    # load for finetune
    last_step = None
    filepath = os.path.join(train_config.checkpoint_path, 'checkpoint_last.pth')
    if args.resume and os.path.isfile(filepath):
        state = torch.load(filepath, map_location=train_config.device)
        generator.load_state_dict(state['generator'])
        mpd.load_state_dict(state['mpd'])
        msd.load_state_dict(state['msd'])
        last_step = state['current_step']

    lr, b1, b2 = train_config.learning_rate, train_config.beta1, train_config.beta2
    optim_g = torch.optim.AdamW(generator.parameters(), lr, betas=[b1, b2])
    optim_d = torch.optim.AdamW(itertools.chain(msd.parameters(), mpd.parameters()), lr, betas=[b1, b2])

    lr_decay = train_config.lr_decay
    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=lr_decay, last_epoch=-1)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=lr_decay, last_epoch=-1)

    logger = WanDBWriter(train_config)

    current_step = 0
    if last_step is not None:
        tqdm_bar = tqdm(total=train_config.epochs * len(training_loader) - last_step)
    else:
        tqdm_bar = tqdm(total=train_config.epochs * len(training_loader))
    validate_func = validate()

    generator.train()
    mpd.train()
    msd.train()
    try:
        for epoch in range(train_config.epochs):
            # dry run for resuming
            if last_step is not None and current_step < last_step:
                for i in range(len(training_loader)):
                    current_step += 1
                    if current_step >= last_step:
                        break
                else:
                    scheduler_g.step()
                    scheduler_d.step()
                    continue
            
            print(f'current epoch = {epoch}')
            for i, batch in enumerate(training_loader):
                current_step += 1
                tqdm_bar.update(1)
                logger.set_step(current_step)

                # Load batch
                target = batch["audio"].float().to(train_config.device)
                target_mel = mel_spec(target)

                # Predict
                pred_wav = generator(target_mel)
                pred_mel = mel_spec(pred_wav)
                target = F.pad(target, (0, pred_wav.size(-1) - target.size(-1)), value=0).unsqueeze(1)

                # Discriminators train
                optim_d.zero_grad()

                y_rs, y_gs, _, _ = mpd(target, pred_wav.detach())
                loss_disc_f = discriminator_loss(y_rs, y_gs)

                y_rs, y_gs, _, _ = msd(target, pred_wav.detach())
                loss_disc_s = discriminator_loss(y_rs, y_gs)

                loss_disc_all = loss_disc_s + loss_disc_f

                loss_disc_all.backward()
                optim_d.step()

                # Generator train
                optim_g.zero_grad()

                target_mel = F.pad(target_mel, (0, pred_mel.size(-1) - target_mel.size(-1)), value=MEL_PAD_VAL).unsqueeze(1)
                loss_mel = F.l1_loss(target_mel, pred_mel) * 45

                _, y_gs, fmap_rs, fmap_gs = mpd(target, pred_wav)
                _, y_hat_gs, fmap_s_rs, fmap_s_gs = msd(target, pred_wav)

                loss_fm_f = feature_loss(fmap_rs, fmap_gs)
                loss_fm_s = feature_loss(fmap_s_rs, fmap_s_gs)

                loss_gen_f = generator_loss(y_gs)
                loss_gen_s = generator_loss(y_hat_gs)
                loss_gen_all = loss_gen_s + loss_gen_f + loss_fm_s + loss_fm_f + loss_mel

                loss_gen_all.backward()
                optim_g.step()

                # Logging
                logger.add_scalar("loss_gen_all", loss_gen_all.item())
                logger.add_scalar("loss_disc_all", loss_disc_all.item())

                logger.add_scalar("loss_gen_s", loss_gen_s.item())
                logger.add_scalar("loss_gen_f", loss_gen_f.item())
                logger.add_scalar("loss_fm_s", loss_fm_s.item())
                logger.add_scalar("loss_fm_f", loss_fm_f.item())
                logger.add_scalar("loss_mel", loss_mel.item())

                logger.add_scalar("loss_disc_s", loss_disc_s.item())
                logger.add_scalar("loss_disc_f", loss_disc_f.item())

                if current_step % train_config.save_step == 0 or current_step == 1:
                    logger.add_image('True spec', target_mel[0].detach().cpu().numpy().T)
                    logger.add_image('Pred spec', pred_mel[0].detach().cpu().numpy().T)
                    logger.set_step(current_step, mode='eval')

                    filename = 'checkpoint_last.pth'
                    os.makedirs(train_config.checkpoint_path, exist_ok=True)
                    torch.save({'generator': generator.state_dict(), 'mpd': mpd.state_dict(), 'msd': msd.state_dict(),
                                'current_step': current_step}, os.path.join(train_config.checkpoint_path, filename))

                    validate_func(os.path.join(train_config.checkpoint_path, filename), logger, mel_spec,
                                  train_config.device)
                    print("save model at step %d ..." % current_step)
            scheduler_g.step()
            scheduler_d.step()
    except KeyboardInterrupt:
        logger.wandb.finish()


if __name__ == '__main__':
    args = argparse.ArgumentParser(description="HiFi-GAN Project")
    args.add_argument(
        "-r",
        "--resume",
        default=True,
        type=bool,
        help="Continues training from the last checkpoint",
    )
    args = args.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    main(args)
