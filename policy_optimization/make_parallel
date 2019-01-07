#!/bin/bash

for reg in -3 -2 -1 0 1 2 3
do
	python gaussian_policy_sim.py -estimator CME -reg_pow ${reg}  &> log.cme.${reg} &
done

python gaussian_policy_sim.py -estimator Direct &> log.cme.${reg} &