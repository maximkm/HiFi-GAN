# HiFi-GAN project
The project is made for educational purposes, as the homework of the course [deep learning for audio processing](https://github.com/markovka17/dla).

## Installation guide
It is recommended to use python 3.8 or 3.9

You need to clone the repository and install the libraries:
```shell
git clone https://github.com/maximkm/HiFi-GAN.git
cd HiFi-GAN
pip install -r requirements.txt
```

## Repository Structure

```bash
inference_data/                     # Examples of possible files for inference
src/
├─ config.py                        # Configs of models
├─ dataloader.py                    # Dataset and dataloader
├─ losses.py                        # Loss function for models
├─ model.py                         # Neural network architectures
├─ utils.py                         # Generating a mel spectrogram from audio
├─ wandb_writer.py                  # Wandb logger
inference.py                     # Script for wav generation
train.py                         # Model Training script
```

## Inference

For inference, you need to download the checkpoint of the pre-trained model, it can be done like this:

```bash
bash prepare_inference.sh
```

For the inference itself, it is enough to run the script `inference.py` to view all the startup arguments, you need to run it with the argument `--help`

A example of running a script:

```bash
python inference.py
```

In this example, the script will take all the files from the `inference_data` folder, and then convert them to wav and save them to the `results` folder

## Reproducing learning

To train the model, you will need to download the LJSpeech dataset. All this can be done with an automated script:

```bash
bash prepare_train.sh
```

Finally, to start the training, it is enough to run the script:

```bash
python train.py
```

More details about training and experiments are written in the report: [Wandb report](https://wandb.ai/maximkm/HiFi-GAN/reports/-HiFi-GAN--VmlldzozMTk3MDI1?accessToken=ers5ldi16ww59pse3833ojpp4u5lvvdye56gw0kgfeqojv77shsfvxj4osn6yeqk)

## Credits

Configs, as well as some implementation details, were peeked at in the official implementation [hifi-gan](https://github.com/jik876/hifi-gan).
