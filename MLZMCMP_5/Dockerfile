FROM python:3.13-slim

RUN pip install pipenv

WORKDIR /app
COPY Pipfile Pipfile.lock ./

RUN pipenv install --system  --deploy

COPY predict.py dv_conv.pkl model_conv.pkl ./

EXPOSE 9696

ENTRYPOINT ["waitress-serve", "--host=0.0.0.0", "--port=9696", "predict:app"]
