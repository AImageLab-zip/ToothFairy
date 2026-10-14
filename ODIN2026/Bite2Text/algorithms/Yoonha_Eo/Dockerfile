FROM python:3.11-slim

# LEGACY SYNTHETIC DEMO ONLY. Actual final2: docker build -t bitemark:final2 submission

WORKDIR /opt/bitemark
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
COPY configs ./configs
RUN python -m pip install --no-cache-dir .

ENTRYPOINT ["bitemark"]

# This public image contains no models and is not a challenge submission image.
# NEEDS TO BE FINETUNED and integrated with a private inference backend.
