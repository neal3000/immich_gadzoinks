# gadzoinks/Dockerfile
FROM python:3.12-slim

WORKDIR /app

# onnxruntime, pillow and numpy are for the tagger. They are the bulk of the
# image (~250MB); the sidecar itself needs none of them, so if you ever split
# tagging into its own container this is the line that moves.
RUN pip install --no-cache-dir \
        fastapi uvicorn psycopg2-binary requests \
        onnxruntime pillow numpy pyyaml huggingface-hub

COPY main.py .
COPY EmbeddedMetadata.py .
COPY WorkflowParser4.py .

# Tagging. Prefixed module names: a plain `vocabulary.py` has already been
# shadowed once by a PyPI package of the same name, and in a flat app
# directory that collision is permanent and silent.
COPY gz_tagger.py .
COPY gz_vocabulary.py .
COPY gz_taxonomy.py .
COPY gz_prompts.py .
COPY gz_taggers.py .
COPY gz_vocabulary.yaml .

# Model cache. Mount a volume here or the ~400MB model is downloaded again
# on every rebuild; the loader checks this cache before touching the network,
# so once it is warm a Hugging Face outage cannot stall a pass.
ENV HF_HOME=/models
VOLUME ["/models"]

EXPOSE 2289
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "2289"]
