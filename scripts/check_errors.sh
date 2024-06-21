#!/bin/bash

# ./check_errors.sh > june_cpu_results_01/errors.out 2>&1 &

res_dir="."

echo ""
echo "--------- Errors -> grep -r Error ---------"
grep -r Error ${res_dir}/results_*/server*
echo ""
echo "--------- Exceptions -> grep -r Exception ---------"
grep -r Exception ${res_dir}/results_*
echo ""
echo "--------- Verify Using CUDA -> grep -r No CUDA ---------"
grep -r "No CUDA" ${res_dir}/results_*
echo ""
echo "--------- Verify ExecutionProvider -> grep -r EP and R using:--------"
grep -r "EP and R using:" ${res_dir}/results_*