FROM odin2026-bite2text-ptv3-debug-v3

ENV BITE2TEXT_RETRIEVAL_INDEX=/opt/ml/model/retrieval_index.npz \
    BITE2TEXT_RETRIEVAL_REPORTS=/opt/ml/model/retrieval_reports.json \
    BITE2TEXT_RETRIEVAL_LABELS=/opt/ml/model/retrieval_labels.json \
    BITE2TEXT_CONFIG=/opt/ml/model/config.py \
    BITE2TEXT_CHECKPOINT=/opt/ml/model/model_final.pth \
    BITE2TEXT_HEAD_VOCABS=/opt/ml/model/head_vocabs.json \
    BITE2TEXT_HYBRID_TOP_K=50 \
    BITE2TEXT_HYBRID_LAMBDA=0.5

LABEL gc_upload_version="2026-08-11-bite2text-ptv3-retrieval-hybrid-v5"

COPY retrieval_inference.py hybrid_inference.py /opt/app/

ENTRYPOINT ["python", "hybrid_inference.py"]
