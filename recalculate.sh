#!/usr/bin/env bash

while true; do
  jupyter nbconvert --to notebook --inplace --execute --allow-errors ./lets-go-baby.ipynb
  sleep 60
done

