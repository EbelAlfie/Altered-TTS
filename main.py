import websockets as ws
import asyncio 
from WhisperAi import Ayaka
from ttsMain import TTS as TextToSpeech

#terima audio binary/ blob
#process jadi text
#dapetin response dari chatgpt beserta tts (stream)
messages = [{'role': 'system', 'content': "You are a reliable assistant"}]
async def startServer():
    async with ws.serve(onConnect, "localhost", "3002"):
        await asyncio.Future()

async def onConnect(websocket):
    message = await websocket.recv() #Selalu dalam bentuk audio binary

    inputText = ayaka.speechToText(message)
    # response = ayaka.getResponse(inputText)
    await getResponse(inputText, websocket)
    

async def getResponse(text: str, websocket):
    messages.append([{'role': 'user', 'content': text}])
    message: str = ""
    response = ayaka.getOpenAi().ChatCompletion.create(
        model='gpt-3.5-turbo',
        messages=messages,
        temperature=0,
        stream=True  
    )
    token:str = ""
    for tokens in response: 
        if tokens['choices'][0]['finish_reason'] != None:
            return 
        message = message + tokens['choices'][0]['delta']['content']
        if tokens['choices'][0]['delta']['content'].startswith(" ") == False:
            token = appendToken(token, token + tokens['choices'][0]['delta']['content'])
        else :
            result = await getVoiceReply(token)
            token = tokens['choices'][0]['delta']['content'].removeprefix(" ")
            await websocket.send(result[0])
            await websocket.send(result[1])
            asyncio.sleep(0.5)
    messages.append([{'role': 'assistant', 'content': message}])
            
def appendToken(word: str, subword: str) -> str:
    return word + subword

async def getVoiceReply(word):
    audio = tts.getVoice(word)
    return (word, audio)

if __name__ == "__main__":
    ayaka = Ayaka()
    tts = TextToSpeech()
    asyncio.run(startServer())