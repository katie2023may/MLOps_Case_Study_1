# Start from the official Prometheus image
FROM prom/prometheus:latest

# Set the working directory
WORKDIR /etc/prometheus

# Copy your Prometheus configuration file into the container
COPY prometheus.yml /etc/prometheus/prometheus.yml

# Expose the Prometheus web interface port
EXPOSE 9090

# Run Prometheus with your configuration file
CMD ["--config.file=/etc/prometheus/prometheus.yml"]





