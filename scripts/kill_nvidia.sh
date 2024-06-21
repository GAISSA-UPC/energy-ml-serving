#!/bin/bash
# Identify processes using /dev/nvidia* devices
PROCESSES=$(fuser -k /dev/nvidia*)

# Check if any processes were found
if [[ ! -z "$PROCESSES" ]]; then
  # Terminate processes with SIGTERM
  kill -SIGTERM $PROCESSES
  echo "Terminated processes using the GPU: $PROCESSES"
fi