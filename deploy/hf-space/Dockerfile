# PhishGuard URL model on a Hugging Face Docker Space (free CPU basic).
# Build context is the staging folder written by scripts/deploy_hf_space.py,
# not the repo root: it holds only this file, requirements.txt, README.md,
# src/phishguard/ and the four v0.2 URL artifacts under models/checkpoints/.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

# Spaces run the container as UID 1000 (huggingface.co/docs/hub/spaces-sdks-docker#permissions).
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONPATH=/home/user/app/src \
    TLDEXTRACT_CACHE=/home/user/.cache/tldextract \
    PHISHGUARD_STORE_RAW_URL=0
WORKDIR /home/user/app

COPY --chown=user requirements.txt ./
RUN pip install --user -r requirements.txt

COPY --chown=user src/ ./src/
COPY --chown=user models/ ./models/
# Read by /healthz to report which source commit is serving.
COPY --chown=user ARTIFACTS.sha256 ./

# Bake the public suffix list into the image at build time so serving never
# depends on a network fetch and every request in one image sees the same list.
# The cache lives under $HOME (owned by user); WORKDIR is created root-owned.
# The test -n step fails the build if nothing was cached.
RUN python -c "import tldextract; print(tldextract.extract('http://a.b.example.co.uk/x'))" \
    && test -n "$(find "$TLDEXTRACT_CACHE" -name '*.json' -print -quit)"

EXPOSE 7860
# tldextract takes a lock file even to read its cache, which fails on a
# read-only root filesystem (the home-server run uses --read-only). Copy the
# baked list to /tmp at start and point tldextract there; this works on a
# writable filesystem too.
CMD ["sh", "-c", "cp -r \"$TLDEXTRACT_CACHE\" /tmp/tldextract && export TLDEXTRACT_CACHE=/tmp/tldextract && exec uvicorn phishguard.serving.api:app --host 0.0.0.0 --port 7860 --no-server-header"]
