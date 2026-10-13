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
    BITE2TEXT_MIDLINE_THRESHOLD=0.45

LABEL gc_upload_version="2026-08-12-bite2text-ptv3-photo-midline-corrected-v7"

COPY retrieval_inference.py photo_inference.py hybrid_inference.py /opt/app/

ENTRYPOINT ["python", "hybrid_inference.py"]
