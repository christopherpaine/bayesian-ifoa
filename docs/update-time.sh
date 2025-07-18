#!/bin/bash

# Path to your HTML file
HTML_FILE="/home/chris-jakoolit/bayesian-ifoa/docs/index.html"

# Get current date and time in desired format
NEW_TIMESTAMP=$(date +"%A %dth %H:%M%P")

# Use sed to replace the old timestamp
sed -i -E "s|(last updated ).* </p>|\1$NEW_TIMESTAMP </p>|" "$HTML_FILE"

