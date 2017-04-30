#!/bin/bash

python clarity_classifier.py
python conciseness_classifier.py

cd submission && zip -r submission.zip *.predict && cd ..
