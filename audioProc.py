from typing import Literal
import warnings
import numpy as np
import wave
import io

temp_files: set[str] = set()
DEFAULT_TEMP_DIR = "../Project Ai/static/Assets/raw/"
format: Literal["wav", "mp3"] = "wav"

def audioBlobToWav(audio) :
    audioStream = io.BytesIO(audio)
    with open('PythonServer\config\input.wav', 'wb') as file:
        file.write(audioStream.read())
    audioStream.close()
    return open('PythonServer\config\input.wav', 'rb')

def processAudio(audioTuple):
    sample_rate, data = audioTuple
    audioBinary = audioToBinary(data, sample_rate, format)
    return audioBinary

def audioToBinary(data: np.ndarray, sampleRate, format):
    if format == "wav":
        data = convert_to_16_bit_wav(data)
    with io.BytesIO() as wavStream:
        channels = (1 if len(data.shape) == 1 else data.shape[1])
        sampleWidth = data.dtype.itemsize
        frames = data.shape[0]
        compType = 'NONE'
        compName = 'not compressed'
        with wave.open(wavStream, 'wb') as wavAudio:
            wavAudio.setnchannels(channels)
            wavAudio.setsampwidth(sampleWidth)
            wavAudio.setframerate(sampleRate)
            wavAudio.setnframes(frames)
            wavAudio.setcomptype(compType, compName)

            wavAudio.writeframes(data.tobytes())

        audioBinary = wavStream.getvalue()
    return audioBinary

def convert_to_16_bit_wav(data):
    # Based on: https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.wavfile.write.html
    warning = "Trying to convert audio automatically from {} to 16-bit int format."
    if data.dtype in [np.float64, np.float32, np.float16]:
        warnings.warn(warning.format(data.dtype))
        data = data / np.abs(data).max()
        data = data * 32767
        data = data.astype(np.int16)
    elif data.dtype == np.int32:
        warnings.warn(warning.format(data.dtype))
        data = data / 65538
        data = data.astype(np.int16)
    elif data.dtype == np.int16:
        pass
    elif data.dtype == np.uint16:
        warnings.warn(warning.format(data.dtype))
        data = data - 32768
        data = data.astype(np.int16)
    elif data.dtype == np.uint8:
        warnings.warn(warning.format(data.dtype))
        data = data * 257 - 32768
        data = data.astype(np.int16)
    else:
        raise ValueError(
            "Audio data cannot be converted automatically from "
            f"{data.dtype} to 16-bit int format."
        )
    return data