#!/usr/bin/env bash

NOTEBOOK="lets-go-baby.ipynb"
BASENAME="${NOTEBOOK%.ipynb}"

echo "Watching ${BASENAME}.py for changes..."
echo "Output: ${BASENAME}.html"

while true; do
  inotifywait -e close_write "${BASENAME}.py" "${BASENAME}.md"
  
  echo "Syncing formats..."
  jupytext --sync "${NOTEBOOK}"

  echo "Recalculating notebook"
  jupyter nbconvert --to notebook --inplace --execute --allow-errors ./lets-go-baby.ipynb

  echo "Rendering HTML..."
  jupyter nbconvert --to html "${NOTEBOOK}" --output "${BASENAME}.html"
  jupyter nbconvert --to html --no-input --no-prompt "${NOTEBOOK}" --output "${BASENAME}-clean.html"



  echo "Done. Waiting for next change..."
done

