FROM odin2026-bite2text-hybrid-debug-v5

ENV BITE2TEXT_RETRIEVAL_INDEX=/opt/ml/model/retrieval_index.npz \
    BITE2TEXT_RETRIEVAL_REPORTS=/opt/ml/model/retrieval_reports.json \
    BITE2TEXT_RETRIEVAL_LABELS=/opt/ml/model/retrieval_labels.json \
    BITE2TEXT_CONFIG=/opt/ml/model/config.py \
    BITE2TEXT_CHECKPOINT=/opt/ml/model/model_final.pth \
    BITE2TEXT_HEAD_VOCABS=/opt/ml/model/head_vocabs.json \
    BITE2TEXT_PHOTO_CHECKPOINT=/opt/ml/model/photo_model_final.pt \
    BITE2TEXT_VIEW_CHECKPOINT=/opt/ml/model/photo_view_classifier.pt \
    BITE2TEXT_HYBRID_TOP_K=50 \
    BITE2TEXT_HYBRID_LAMBDA=0.5 \
    BITE2TEXT_PHOTO_LAMBDA=0.2 \
    BITE2TEXT_MIDLINE_THRESHOLD=0.45 \
    BITE2TEXT_PRECISION_SANITIZER=1 \
    BITE2TEXT_RISK_RERANK=1 \
    BITE2TEXT_RISK_MARGIN=0.02 \
    BITE2TEXT_UNSUPPORTED_PENALTY=0.005 \
    BITE2TEXT_UNSUPPORTED_GATE=5 \
    BITE2TEXT_CONTRADICTION_THRESHOLD=0.65 \
    BITE2TEXT_CONTRADICTION_GATE=0.01 \
    BITE2TEXT_CONTRADICTION_PENALTY=0.5 \
    BITE2TEXT_MIN_CONTRADICTION_IMPROVEMENT=0.015 \
    BITE2TEXT_NO_NEW_UNSUPPORTED=1

LABEL gc_upload_version="2026-08-18-bite2text-risk-rerank-v9"

COPY retrieval_inference.py photo_inference.py report_sanitizer.py hybrid_inference.py /opt/app/

ENTRYPOINT ["python", "hybrid_inference.py"]
