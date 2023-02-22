from fastapi import FastAPI, Response

from tts1.kazakh_tts_service import KazakhTtsService

app = FastAPI()


@app.get("/api/text2speach", response_class=Response, response_model=bytes)
async def text2speach(text: str):
    wav_bytes = KazakhTtsService().text2speach_bytes(text)
    headers = {'Content-Disposition': 'attachment; filename="speach.wav"'}
    return Response(wav_bytes, headers=headers, media_type='audio/wav')
