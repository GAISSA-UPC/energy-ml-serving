#!/bin/bash

# ./scripts/check_errors.sh > nov_results_01/nov_exp_02/errors.out 2>&1 &
# ./scripts/check_errors.sh > errors.out 2>&1 &

#res_dir="./gpu_june_phi2_01/"
res_dir="." #/nov_results_01/nov_exp_02

echo ""
echo "--------- Errors -> grep -r Error ---------"
grep -ri error ${res_dir}/results_*/server*
grep -ri error ${res_dir}/results/server*

echo ""
echo "--------- Exceptions -> grep -r Exception ---------"
grep -ri exception ${res_dir}/results_*
grep -ri exception ${res_dir}/results

echo ""
echo "--------- Verify Using CUDA -> grep -r No CUDA ---------"
grep -ri "No CUDA" ${res_dir}/results_*
grep -ri "No CUDA" ${res_dir}/results

echo ""
echo "--------- Verify ExecutionProvider -> grep -r EP and R using:--------"
grep -r "EP and R using:" ${res_dir}/results_*
grep -r "EP and R using:" ${res_dir}/results
