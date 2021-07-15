FROM python:3.8

RUN pip install virtualenv
ENV VIRTUAL_ENV=/venv
RUN virtualenv venv -p python3
ENV PATH="VIRTUAL_ENV/bin:$PATH"

WORKDIR /app
ADD . /app

# Install dependencies
RUN pip install -r requirements.txt
RUN pip install --upgrade nltk
RUN python3 -m nltk.downloader punkt
RUN python3 -m nltk.downloader wordnet

# Expose port
EXPOSE 8080

# Run the application:
CMD ["python", "app/application.py"]
