#!/bin/bash

MODE=$1
if [ -z "$MODE" ]; then
  MODE="top"
elif [[ "$MODE" != "top" && "$MODE" != "new" && "$MODE" != "hot" ]]; then
  MODE="top"
fi
LABEL_ARG=$2
if [ -z "$LABEL_ARG" ]; then
  LABEL_ARG="manual"
elif [[ "$LABEL_ARG" != "manual" && "$LABEL_ARG" != "automatic" ]]; then
  LABEL_ARG="manual"
fi

python3 code/data_collect.py $MODE

python3 code/preprocessing.py

python3 code/modeling.py

python3 code/topic_labeling_ai.py $LABEL_ARG

python3 code/analysis.py

echo "Done!"
