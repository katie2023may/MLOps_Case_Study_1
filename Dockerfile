FROM python:3.13-slim

WORKDIR /opt/app

COPY . .

# Upgrade pip to the latest version
RUN pip install --upgrade pip

# Install Python dependencies
RUN pip install --no-cache-dir -r /opt/app/requirements.txt

# Install packages that we need. vim is for helping with debugging
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get upgrade -yq ca-certificates && \
    apt-get install -yq --no-install-recommends \
    prometheus-node-exporter

EXPOSE 7860
EXPOSE 8000
EXPOSE 9100
ENV GRADIO_SERVER_NAME="0.0.0.0"

CMD bash -c "prometheus-node-exporter --web.listen-address=':9100' & python /opt/app/app.py"