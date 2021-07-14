FROM python:3.7.6

RUN pip install virtualenv
ENV VIRTUAL_ENV=venv
RUN virtualenv venv -p python3
ENV PATH="VIRTUAL_ENV/bin:$PATH"

WORKDIR /app
ADD . /app

# Install dependencies
RUN pip install -r requirements.txt
RUN python3 -m nltk.downloader punkt
RUN python3 -m nltk.downloader wordnet

# Expose port
EXPOSE 80

# Run the application:
CMD ["python", "app/run.py"]
