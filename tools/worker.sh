#!/bin/bash
# worker.sh JOBS GPU : run the jobs of JOBS (one per line: NAME<TAB>NEEDS<TAB>COMMAND) that nobody has taken yet, in order, on GPU.
# A job is taken by creating logs/locks/NAME, finished by logs/done/NAME (or logs/failed/NAME); the file is read again for
# every job, so jobs can be appended while workers run, and more workers (on more GPUs) can be started at any time.
# A job with NEEDS other than "-" waits until logs/done/NEEDS exists. Run from the repository root.
JOBS=$1; GPU=$2
mkdir -p logs/locks logs/done logs/failed logs/jobs logs/geom
export CUDA_VISIBLE_DEVICES=$GPU PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
while true; do
  PICK=""
  while IFS=$'\t' read -r NAME NEEDS CMD; do
    [ -z "$NAME" ] && continue
    [ -e logs/done/$NAME ] || [ -e logs/failed/$NAME ] && continue
    [ "$NEEDS" != "-" ] && [ ! -e logs/done/$NEEDS ] && continue
    if mkdir logs/locks/$NAME 2>/dev/null; then PICK="$NAME"; PCMD="$CMD"; break; fi
  done < <(cat $JOBS)
  [ -z "$PICK" ] && break
  if bash -c "$PCMD" > logs/jobs/$PICK.log 2>&1; then touch logs/done/$PICK; else touch logs/failed/$PICK; fi
  rmdir logs/locks/$PICK
done
