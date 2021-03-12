#!/bin/bash -eu

rm -rf *.prof legion_prof
export LG_RT_DIR=~/legion/runtime
srun --exclusive -N 1 bash -c '. ~/soleil-x/src/sapling_reload_modules.sh; ./backpressure -ll:cpu 1 -ll:csize 1 -lg:prof 1 -level app=2,mapper=2 -lg:prof_logfile %.prof -lg:debug_ok'
~/legion/tools/legion_prof.py *.prof
