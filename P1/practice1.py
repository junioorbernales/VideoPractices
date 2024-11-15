'''
%pip3 install fastapi
%pip3 install "uvicorn[standard]" #Servidor virtual para usar FastAPI
'''

from fastapi import FastAPI
app = FastAPI()
@app.get("/")
async def root():
 return {"greeting":"Hello world"}