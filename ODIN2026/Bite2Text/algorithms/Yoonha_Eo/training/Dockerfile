FROM bitemark:final2
USER root
RUN python -m pip install --no-cache-dir iterative-stratification==0.1.9 wandb==0.19.11
COPY training /workspace/training
COPY final2/model /workspace/final2/model
ENV PYTHONPATH=/workspace/final2/model/s/a:/workspace/final2/model/z \
    WANDB_MODE=disabled
WORKDIR /workspace
USER algorithm
ENTRYPOINT ["python"]
