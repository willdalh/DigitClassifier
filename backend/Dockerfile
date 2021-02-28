FROM python:3.6.10

RUN mkdir -p /usr/src/app
WORKDIR /usr/src/app

COPY requirements.txt /usr/src/app
# COPY gunicorn_config.py /usr/src/app
RUN pip install -r requirements.txt --no-cache-dir
CMD ["ls"]
# ADD . /usr/src/app
# EXPOSE 5000
# RUN chmod +x ./entrypoint.sh
# ENTRYPOINT ["sh", "entrypoint.sh"]
ENTRYPOINT [ "flask" ]
CMD ["run", "--host=0.0.0.0", "--port=5000"]