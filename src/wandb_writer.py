from datetime import datetime
import wandb


class WanDBWriter:
    def __init__(self, config):
        self.writer = None
        self.selected_module = ""

        wandb.login()

        if not hasattr(config, 'wandb_project'):
            raise ValueError("please specify project name for wandb")

        wandb.init(
            project=getattr(config, 'wandb_project'),
            config=config
        )
        self.wandb = wandb

        self.step = 0
        self.mode = ""
        self.timer = datetime.now()

    def set_step(self, step, mode="train"):
        self.mode = mode
        self.step = step
        if step == 0:
            self.timer = datetime.now()
        else:
            duration = datetime.now() - self.timer
            self.add_scalar("steps_per_sec", 1 / duration.total_seconds())
            self.timer = datetime.now()

    def scalar_name(self, scalar_name):
        return f"{self.mode}/{scalar_name}"

    def add_scalar(self, scalar_name, scalar):
        self.wandb.log({
            self.scalar_name(scalar_name): scalar,
        }, step=self.step)

    def add_scalars(self, tag, scalars):
        self.wandb.log({
            **{f"{scalar_name}_{tag}_{self.mode}": scalar for scalar_name, scalar in scalars.items()}
        }, step=self.step)

    def add_image(self, scalar_name, image):
        self.wandb.log({
            self.scalar_name(scalar_name): self.wandb.Image(image)
        }, step=self.step)

    def add_audio(self, scalar_name, audio, sample_rate=None):
        self.wandb.log({
            self.scalar_name(scalar_name): self.wandb.Audio(audio, sample_rate=sample_rate)
        }, step=self.step)
