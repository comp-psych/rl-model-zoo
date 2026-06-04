#!/bin/bash
cd /Users/peterzhou/Research/Labs/NSG/comp-psych-org/rl-model-zoo
python3 -m pytest tests/ -v --tb=short 2>&1
