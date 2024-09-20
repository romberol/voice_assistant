import numpy as np
import torch
torch.set_num_threads(1)
import torchaudio
torchaudio.set_audio_backend("soundfile")
import pyaudio
import soundfile as sf
import requests
import string
import os


model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                              model='silero_vad')

CONFIG = {
    "silence_threshold": 0.1,    # VAD confidence threshold for silence
    "silence_duration": 1.5,     # seconds of silence to detect the end of speech
    "min_speech_duration": 2.0,  # Minimum speech duration in seconds to save
    "max_speech_duration": 30.0, # Maximum speech duration in seconds
    "num_samples": 512
}


class SileroVAD:
    def __init__(self, config):
        self.model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad')
        self.get_speech_timestamps, self.save_audio, self.read_audio, self.VADIterator, self.collect_chunks = utils

        self.SAMPLE_RATE = 16000
        self.audio = pyaudio.PyAudio()

        self.silence_threshold = config["silence_threshold"]
        self.num_samples = config["num_samples"]
        self.api_url = "http://127.0.0.1:5000"

        self.speech_buffer = []
        self.is_speaking = False
        self.silence_counter = 0
        self.silence_chunks = int(config["silence_duration"] * self.SAMPLE_RATE / self.num_samples)
        self.min_speech_chunks = int(config["min_speech_duration"] * self.SAMPLE_RATE / self.num_samples)
        self.max_speech_chunks = int(config["max_speech_duration"] * self.SAMPLE_RATE / self.num_samples)

        self.stream = None
        os.makedirs("detected_speech", exist_ok=True)


    def validate(self, inputs: torch.Tensor):
        with torch.no_grad():
            outs = self.model(inputs)
        return outs

    def int2float(self, sound):
        abs_max = np.abs(sound).max()
        sound = sound.astype('float32')
        if abs_max > 0:
            sound *= 1/32768
        sound = sound.squeeze()
        return sound

    def start_stream(self):
        self.stream = self.audio.open(format=pyaudio.paInt16,
                                      channels=1,
                                      rate=self.SAMPLE_RATE,
                                      input=True,
                                      frames_per_buffer=int(self.SAMPLE_RATE / 10))
        print("Started Recording")

    def prepare_prompt(self, transcription):
        # remove punctuation
        translator = str.maketrans('', '', string.punctuation)
        cleaned_transcription = transcription.lower().translate(translator)
        # search for trigger word
        if "orange" in cleaned_transcription.split(): #[:2]
            index = cleaned_transcription.find("orange")
            return cleaned_transcription[index + len("orange"):]
        return None

    def transcribe(self, filename):
        payload = {"file_name": filename}
        response = requests.post(self.api_url + "/transcribe", json=payload)
        if response.status_code != 200:
            print(f"Failed to transcribe. Status code: {response.status_code}")
            print("Response:", response.text)
        return response.json()["transcription"]

    def ask_llm(self, prompt):
        payload = {'prompt': prompt + " </s> <|assistant|>"}

        response = requests.post(self.api_url + "/answer", json=payload, stream=True)

        if response.status_code == 200:
            for chunk in response.iter_content(chunk_size=128):
                if chunk:
                    print(chunk.decode('utf-8'), end='')
            print("\n====== End of response ======")
        else:
            print(f"Request failed with status code {response.status_code}: {response.text}")

    def process_audio(self, max_iterations=1000):
        i = 0
        while True:
            i += 1
            audio_chunk = self.stream.read(self.num_samples)

            # convert audio to int16 and then to float32
            audio_int16 = np.frombuffer(audio_chunk, np.int16)
            audio_float32 = self.int2float(audio_int16)

            new_confidence = self.model(torch.from_numpy(audio_float32), self.SAMPLE_RATE).item()
            if self.is_speaking:
                self.speech_buffer.append(audio_float32)
            if new_confidence > self.silence_threshold:
                if not self.is_speaking: print("Start of speech detected")
                self.is_speaking = True
                self.silence_counter = 0
            elif self.is_speaking:
                # no speech detected
                self.silence_counter += 1
                if self.silence_counter > self.silence_chunks or len(self.speech_buffer) > self.max_speech_chunks:
                    # end of speech detected
                    if len(self.speech_buffer) > self.min_speech_chunks:
                        print("End of speech detected. Sending audio for transcription.")
                        file_path = f'detected_speech/speech_segment{i}.wav'
                        waveform = np.concatenate(self.speech_buffer)
                        waveform /= np.max(np.abs(waveform)) + 1e-9
                        sf.write(file_path, waveform, self.SAMPLE_RATE)
                        transcription = self.transcribe(file_path)
                        prompt = self.prepare_prompt(transcription)
                        if prompt:
                            print("User:", transcription)
                            self.ask_llm(prompt)
                        else:
                            print("No trigger words detected:", transcription)

                        self.speech_buffer = []  # clear the buffer for the next speech segment
                    self.is_speaking = False

            if max_iterations is not None and i > max_iterations:
                break

    def stop_stream(self):
        self.stream.stop_stream()
        self.stream.close()
        self.audio.terminate()
        print("Stopped the recording")


vad = SileroVAD(CONFIG)
vad.start_stream()
vad.process_audio(max_iterations=None)
vad.stop_stream()