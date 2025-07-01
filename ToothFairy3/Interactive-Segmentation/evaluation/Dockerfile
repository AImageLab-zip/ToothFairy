FROM --platform=linux/amd64 python:3.10-slim

ENV PYTHONUNBUFFERED=1

# Install dependencies system-wide as root
WORKDIR /opt/app

COPY requirements.txt /opt/app/

RUN python -m pip install --no-cache-dir --no-color --requirement requirements.txt

# Create non-root user and switch to it *after* installing dependencies
RUN groupadd -r user && useradd -m -r -g user user

USER user

COPY --chown=user:user evaluation.py /opt/app/

ENTRYPOINT ["python", "-m", "evaluation"]

