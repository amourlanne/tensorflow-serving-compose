FROM python:3.6

RUN mkdir /usr/src/app/
COPY ./requirements.txt /usr/src/app/requirements.txt

WORKDIR /usr/src/app/

RUN pip install -r requirements.txt

EXPOSE 5000

CMD ["python", "app.py"]