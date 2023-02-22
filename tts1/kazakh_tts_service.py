import io
import os

import numpy
import scipy
import torch
from espnet2.bin.tts_inference import Text2Speech
from parallel_wavegan.utils import load_model

fs = 22050

## specify the path to vocoder's checkpoint
os.chdir('tts1')
vocoder_checkpoint = "exp/vocoder/checkpoint-400000steps.pkl"
vocoder = load_model(vocoder_checkpoint).to("cpu").eval()
vocoder.remove_weight_norm()

## specify path to the main model(transformer/tacotron2/fastspeech) and its config file
config_file = "exp/tts_train_raw_char/config.yaml"
model_path = "exp/tts_train_raw_char/train.loss.ave_5best.pth"
# config_file = "exp/lm_train_lm_ksc2_char/config.yaml"
# model_path = "exp/lm_train_lm_ksc2_char/valid.acc.ave_10best.pth"


class KazakhTtsService:
    def __init__(self):
        self.text2speech = Text2Speech(
            config_file,
            model_path,
            device="cpu",
            # Only for Tacotron 2
            threshold=0.5,
            minlenratio=0.0,
            maxlenratio=10.0,
            use_att_constraint=True,
            backward_window=1,
            forward_window=3,
            # Only for FastSpeech & FastSpeech2
            speed_control_alpha=1.0,
        )
        self.text2speech.spc2wav = None  # Disable griffin-lim

    def text2speach_bytes(self, text: str) -> bytes:
        with torch.no_grad():
            output_dict = self.text2speech(text.lower())
            feat_gen = output_dict['feat_gen']
            wav: torch.Tensor = vocoder.inference(feat_gen)

        view: torch.Tensor = wav.view(-1)
        cpu_result: torch.Tensor = view.cpu()
        arr: numpy.ndarray = cpu_result.numpy()
        with io.BytesIO() as buf:
            scipy.io.wavfile.write(buf, fs, arr)
            wav_bytes = buf.getvalue()

        return wav_bytes
