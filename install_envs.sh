#!/bin/bash

cd envs
cd metaworld
pip install -e .
cd ..
cd mujoco-control-envs
pip install -e .
cd ../..
