#!/usr/bin/env bash

# Path to the script you want to run
SCRIPT_PATH="/home/chris-jakoolit/bayesian-ifoa/docs/copy-files-across.sh"

# Cron schedule
CRON_LINE="*/15 * * * * $SCRIPT_PATH"

# Get current crontab into a variable
CURRENT_CRON=$(crontab -l 2>/dev/null)

# Check if the line is already in crontab
if echo "$CURRENT_CRON" | grep -Fxq "$CRON_LINE"; then
  echo "Cron job found. Removing..."
  # Remove the line and update crontab
  echo "$CURRENT_CRON" | grep -Fxv "$CRON_LINE" | crontab -
  echo "Cron job removed."
else
  echo "Cron job not found. Adding..."
  # Append the line and update crontab
  (echo "$CURRENT_CRON"; echo "$CRON_LINE") | crontab -
  echo "Cron job added."
fi

