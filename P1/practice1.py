'''
%pip3 install fastapi
%pip3 install "uvicorn[standard]" #Servidor virtual para usar FastAPI
uvicorn P1.practice1:app --reload para encender el servidor
'''

from fastapi import FastAPI

app = FastAPI()

@app.get("/")
async def root():
    return {"greeting":"Hello world"}