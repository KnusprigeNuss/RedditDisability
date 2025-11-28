#!/bin/bash

while [[ "$#" -gt 0 ]]; do
  case $1 in
    --merge)
      python3 code/csv_merger.py
      shift
      ;;
      
    --collect)
      shift
      python3 code/data_collect.py "$@"
      exit 0
      ;;
      
    --preprocess)
      shift
      python3 code/preprocessing.py "$@"
      exit 0
      ;;
      
    --model)
      shift
      python3 code/modeling.py "$@"
      exit 0
      ;;
      
    --labeling)
      shift
      python3 code/topic_labeling_ai.py "$@"
      exit 0
      ;;
      
    --analysis)
      shift
      python3 code/analysis.py "$@"
      exit 0
      ;;
      
    *)
      echo "Unknown flag: $1"
      exit 1
      ;;
  esac
done
