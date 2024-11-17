FROM python:3.11-slim

WORKDIR /P1

COPY ./requirements.txt ./API/requirements.txt 
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY . .

CMD ["fastapi", "run", "P1/practice1.py", "--port", "80"]