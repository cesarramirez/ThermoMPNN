#!/bin/bash
unset PYTHONPATH
unset LD_LIBRARY_PATH
export PYTHONNOUSERSITE=1
export PATH="/opt/conda/conda/envs/thermoMPNN/bin:$PATH"
exec "$@" 