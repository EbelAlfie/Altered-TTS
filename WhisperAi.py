import openai
import audioProc
import utils
from ttsMain import TTS as TextToSpeech

class Ayaka():
    def __init__(self) -> None:
        self.messages = [{'role': 'system', 'content': "You are a very reliable assistant"}]
        openai.api_key = utils.get_hparams_from_file(r'PythonServer\config\config.json').openAiApiKey

    def speechToText(self, audioBlob) -> str:
        audio = audioProc.audioBlobToWav(audioBlob) #todo convert audio binary to wav 
        transcript = openai.Audio.transcribe(model="whisper-1", file=audio)
        return transcript.text

    def getResponse(self, text: str) -> tuple:
        self.messages.append([{'role': 'user', 'content': text}])
        response = openai.ChatCompletion.create(
            model='gpt-3.5-turbo',
            messages= self.messages,
            temperature=0    
        )
        reply = response['choices'][0]['message']['content']
        self.messages.append([{'role': 'assistant', 'content': reply}])
        audio = TextToSpeech().getVoice(reply)
        return (reply, audio)
    def getOpenAi():
        return openai