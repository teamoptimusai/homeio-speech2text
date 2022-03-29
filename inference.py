import pyaudio
import threading
import time
import argparse
import wave
import torchaudio
import torch
from utils.dataset import get_featurizer
# from utils.decoder import DecodeGreedy, CTCBeamDecoder
from utils.decoder import DecodeGreedy, GreedyCTCDecoder
import os

CHUNCK_SIZE = 1024


class Listener:

    def __init__(self, sample_rate=8000, record_seconds=2):
        self.chunk = CHUNCK_SIZE
        self.sample_rate = sample_rate
        self.record_seconds = record_seconds
        self.p = pyaudio.PyAudio()
        if not args.wav_file:
            self.stream = self.p.open(format=pyaudio.paInt16,
                                      channels=1,
                                      rate=self.sample_rate,
                                      input=True,
                                      output=True,
                                      frames_per_buffer=self.chunk)

    def listen(self, queue):
        while True:
            data = self.stream.read(self.chunk, exception_on_overflow=False)
            queue.append(data)
            time.sleep(0.01)

    def run(self, queue):
        thread = threading.Thread(
            target=self.listen, args=(queue,), daemon=True)
        thread.start()
        print("\Speech Recognition Engine is now listening... \n")

    def run_wav(self, wav_file, queue):
        wav_file = wave.open(wav_file, 'rb')
        while True:
            data = wav_file.readframes(self.chunk)
            if data == b'':
                break
            queue.append(data)
            time.sleep(0.01)


class SpeechRecognitionEngine:

    def __init__(self, model_file, ken_lm_file, context_length=10, sample_rate=8000):
        self.sample_rate = sample_rate
        self.listener = Listener(self.sample_rate)
        self.model = torch.jit.load(model_file)
        self.model.eval().to('cpu')  # run on cpu
        self.featurizer = get_featurizer(8000)
        self.audio_q = list()
        self.hidden = (torch.zeros(1, 1, 1024), torch.zeros(1, 1, 1024))
        self.beam_results = ""
        self.out_args = None
        # self.beam_search = CTCBeamDecoder(
        #     beam_size=100, kenlm_path=ken_lm_file)
        # # multiply by 50 because each 50 from output frame is 1 second
        self.GreedyCTCDecoder = GreedyCTCDecoder()
        print(self.GreedyCTCDecoder.labels)
        self.context_length = context_length * 50
        self.start = False
        self.n = 0

    def save(self, waveforms, fname="temp/audio/audio_temp"):
        wf = wave.open(f'{fname}{self.n}.wav', "wb")
        wf.setnchannels(1)
        wf.setsampwidth(self.listener.p.get_sample_size(pyaudio.paInt16))
        wf.setframerate(self.sample_rate)
        wf.writeframes(b"".join(waveforms))
        wf.close()
        return f'{fname}{self.n}.wav'

    def predict(self, audio):
        with torch.no_grad():
            fname = self.save(audio)
            self.n = self.n + 1
            waveform, _ = torchaudio.load(fname)  # don't normalize on train
            log_mel = self.featurizer(waveform).unsqueeze(1)
            out, self.hidden = self.model(log_mel, self.hidden)
            results = DecodeGreedy(out)
            # results = self.GreedyCTCDecoder(out)
            out = torch.nn.functional.softmax(out, dim=2)
            out = out.transpose(0, 1)
            self.out_args = out if self.out_args is None else torch.cat(
                (self.out_args, out), dim=1)
            # decoder_test(self.out_args)
            # results = self.beam_search(self.out_args)
            current_context_length = self.out_args.shape[1] / 50  # in seconds
            if self.out_args.shape[1] > self.context_length:
                self.out_args = None
            return results, current_context_length

    def inference_loop(self, action):
        while True:
            if len(self.audio_q) < 5:
                continue
            else:
                pred_q = self.audio_q.copy()
                self.audio_q.clear()
                action(self.predict(pred_q))
            time.sleep(0.05)

    def run(self, action):
        self.listener.run(self.audio_q)
        thread = threading.Thread(target=self.inference_loop,
                                  args=(action,), daemon=True)
        thread.start()

    def predict_wav(self, queue, fname="temp/audio/audio_temp"):
        prediction = []
        with torch.no_grad():
            for i, audio in enumerate(queue):
                wf = wave.open(f'{fname}{i}.wav', "wb")
                wf.setnchannels(1)
                wf.setsampwidth(
                    self.listener.p.get_sample_size(pyaudio.paInt16))
                wf.setframerate(self.sample_rate)
                wf.writeframes(audio)
                wf.close()
                waveform, _ = torchaudio.load(f'{fname}{i}.wav')
                log_mel = self.featurizer(waveform).unsqueeze(1)
                out, self.hidden = self.model(log_mel, self.hidden)
                results = DecodeGreedy(out)
                prediction.append(results)
        return "".join(prediction)

    def run_wav(self, wav_file):
        self.listener.run_wav(wav_file, self.audio_q)
        print(self.predict_wav(self.audio_q))


class DemoAction:

    def __init__(self):
        self.asr_results = ""
        self.current_beam = ""

    def __call__(self, x):
        results, current_context_length = x
        self.current_beam = results
        trascript = " ".join(self.asr_results.split() + results.split())
        print(trascript)
        if current_context_length > 10:
            self.asr_results = trascript


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="demoing the speech recognition engine in terminal.")
    parser.add_argument('--model_file', type=str, default=None, required=True,
                        help='optimized file to load. use optimize_graph.py')
    parser.add_argument('--ken_lm_file', type=str, default=None, required=False,
                        help='If you have an ngram lm use to decode')
    parser.add_argument('--wav_file', type=str, default=None, required=False)
    args = parser.parse_args()

    os.makedirs("temp/audio", exist_ok=True)

    asr_engine = SpeechRecognitionEngine(args.model_file, args.ken_lm_file)
    if args.wav_file:
        asr_engine.run_wav(args.wav_file)
    else:
        action = DemoAction()
        asr_engine.run(action)
        threading.Event().wait()
