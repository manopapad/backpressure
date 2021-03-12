#!/bin/bash -eu

export LG_RT_DIR=~/legion/runtime
srun --export=all --exclusive -N 1 bash -c '. ~/soleil-x/src/sapling_reload_modules.sh; make -j'
