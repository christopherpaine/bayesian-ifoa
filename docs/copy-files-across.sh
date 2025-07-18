#!/usr/bin/env bash
cd /home/chris-jakoolit/bayesian-ifoa/docs || exit 1

SCRIPT_PATH="/home/chris-jakoolit/bayesian-ifoa/docs"
SCRIPT_PATH2="/home/chris-jakoolit/123/bayesian_inference/chris_pca"

cp "${SCRIPT_PATH2}/lets-go-baby.html" "$SCRIPT_PATH/"
cp "${SCRIPT_PATH2}/lets-go-baby-clean.html" "$SCRIPT_PATH/"
${SCRIPT_PATH}/update-time.sh
git add .
git commit -m "."
git push
