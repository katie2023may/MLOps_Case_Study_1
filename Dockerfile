FROM python:3.13-slim

WORKDIR /opt/app

ENV DEBIAN_FRONTEND=noninteractive
# Install packages that we need. vim is for helping with debugging
RUN apt-get update && \
    apt-get upgrade -yq ca-certificates && \
    apt-get install -yq --no-install-recommends \
    prometheus-node-exporter

COPY requirements.txt /opt/app/requirements.txt
RUN pip install --no-cache-dir -r /opt/app/requirements.txt

COPY . .

EXPOSE 7860
EXPOSE 8000
EXPOSE 9100
ENV GRADIO_SERVER_NAME="0.0.0.0"

CMD bash -c "prometheus-node-exporter --web.listen-address=':9100' & python /opt/app/app.py"