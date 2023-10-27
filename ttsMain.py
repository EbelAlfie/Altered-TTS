# coding=utf-8
import argparse
import audioProc
import utils
import commons
import torch
from models import SynthesizerTrn
from text import text_to_sequence, _clean_text
from torch import no_grad, LongTensor

class TTS():
    def __init__(self) -> None:
        super().__init__()
        hps_ms = utils.get_hparams_from_file(r'PythonServer\config\config.json')
        parser = argparse.ArgumentParser()
        parser.add_argument('--device', type=str, default='cpu')
        parser.add_argument('--api', action="store_true", default=False)
        parser.add_argument("--share", action="store_true", default=False, help="share gradio app")
        args = parser.parse_args()
        device = torch.device(args.device)
        net_g_ms = SynthesizerTrn(
            len(hps_ms.symbols),
            hps_ms.data.filter_length // 2 + 1,
            hps_ms.train.segment_size // hps_ms.data.hop_length,
            n_speakers=hps_ms.data.n_speakers,
            **hps_ms.model)
        utils.load_checkpoint('PythonServer\pretrained_models\\ayaka-jp\\ayaka-jp.pth', net_g_ms, None)
        _ = net_g_ms.eval().to(device)

        self.hps_ms = hps_ms
        self.parser = parser
        self.device = device
        self.net_g_ms = net_g_ms

    def get_text(self, text, hps, is_symbol):
        text_norm, clean_text = text_to_sequence(text, hps.symbols, [] if is_symbol else hps.data.text_cleaners)
        if hps.data.add_blank:
            text_norm = commons.intersperse(text_norm, 0)
        text_norm = LongTensor(text_norm)
        return text_norm, clean_text

    def tts_fn(self, text, language, speaker_id, noise_scale, noise_scale_w, length_scale, is_symbol):
        device = self.device
        hps_ms = self.hps_ms
        net_g_ms = self.net_g_ms

        sampleRate = 22050
        text = text.replace('\n', ' ').replace('\r', '').replace(" ", "")
        if not is_symbol:
            if language == 0:
                text = f"[ZH]{text}[ZH]"
            elif language == 1:
                text = f"[JA]{text}[JA]"
            else:
                text = f"{text}"
        stn_tst, clean_text = self.get_text(text, hps_ms, is_symbol)
        with no_grad():
            x_tst = stn_tst.unsqueeze(0).to(device)
            x_tst_lengths = LongTensor([stn_tst.size(0)]).to(device)
            sid = LongTensor([speaker_id]).to(device)
            audio = net_g_ms.infer(x_tst, x_tst_lengths, sid=sid, noise_scale=noise_scale, noise_scale_w=noise_scale_w,
                                    length_scale=length_scale)[0][0, 0].data.cpu().float().numpy()
        return (sampleRate, audio)
    
    def getVoice(self, text) -> bytes:
        replyVoice = self.generateVoice(text, 303)
        #process audio
        data = audioProc.processAudio(replyVoice)
        return data
        # with open('output_audio.wav', 'wb') as file:
        #     file.write(data) debug

    def generateVoice(self, replyText, speakerId):
        language = 2
        noise_scale = 0.6
        noise_scale_w = 0.668
        length_scale = 1.2
        is_symbol = 1
        return self.tts_fn(replyText, language, speakerId, noise_scale, noise_scale_w, length_scale, is_symbol)