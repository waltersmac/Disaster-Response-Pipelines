FROM python:3.8.10

RUN pip install virtualenv
ENV VIRTUAL_ENV=/venv
RUN virtualenv venv -p python3
ENV PATH="VIRTUAL_ENV/bin:$PATH"

WORKDIR /app
ADD . /app

RUN pip install -r requirements.txt

# Run the application:
CMD ["flask", "run", "--host", "0.0.0.0"]
