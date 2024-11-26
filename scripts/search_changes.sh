#!/bin/bash

# Useful commands to verify needed changes before running experiments
# search for changes, help for experiments
echo "------------------------"
grep -r CHANGE app/*
grep -r ADD app/*

echo "------------------------"
grep -r CHANGE testing/*
grep -r ADD testing/*

echo "------------------------"
grep -r CHANGE *.sh
grep -r ADD *.sh
