FROM odin2026-bite2text-ptv3-debug-v3

ENV BITE2TEXT_RETRIEVAL_INDEX=/opt/ml/model/retrieval_index.npz \
    BITE2TEXT_RETRIEVAL_REPORTS=/opt/ml/model/retrieval_reports.json

LABEL gc_upload_version="2026-08-10-bite2text-geometry-retrieval-v4"

COPY retrieval_inference.py /opt/app/retrieval_inference.py

ENTRYPOINT ["python", "retrieval_inference.py"]

