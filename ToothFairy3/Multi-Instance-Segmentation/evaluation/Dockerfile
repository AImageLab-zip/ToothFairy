FROM --platform=linux/amd64 python:3.11-slim

# Ensures that Python output to stdout/stderr is not buffered
ENV PYTHONUNBUFFERED=1

# Create a non-root user and group
RUN groupadd -r user && useradd -m --no-log-init -r -g user user
USER user

WORKDIR /opt/app

# Copy and install requirements as user
COPY --chown=user:user requirements.txt /opt/app/
RUN python -m pip install \
    --user \
    --no-cache-dir \
    --no-color \
    --requirement /opt/app/requirements.txt

# Copy application code
COPY --chown=user:user evaluation.py /opt/app/

ENTRYPOINT ["python", "-m", "evaluation"]
