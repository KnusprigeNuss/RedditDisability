#!/bin/bash

MODE=$1
if [ -z "$MODE" ]; then
  MODE="top"
fi

echo "📥 Fetching r/disability posts using mode: $MODE"
python3 code/data_collect.py $MODE

echo "🧼 Running preprocessing..."
python3 code/preprocessing.py

echo "🤖 Running topic modeling..."
python3 code/modeling.py

echo "✅ Done!"
