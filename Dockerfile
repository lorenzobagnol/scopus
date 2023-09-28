FROM python:3.10-alpine3.17
RUN pip install --upgrade pip
WORKDIR /SearchApp
COPY . /SearchApp
RUN pip install -r requirements.txt
# EXPOSE port
CMD python ./execute.py 