#!/usr/bin/env bash

PATTERN=.
if [ -n "$1" ]
then
  PATTERN=$1
fi     # $String is null.

for i in $PATTERN; do
    python parse_results_epochs_v1.py $i
done
