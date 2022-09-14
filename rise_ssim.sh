#!/bin/bash
module add python3-3.8.0
source venv/bin/activate bash
which python
python -m Temperature_Scaling.compare_models