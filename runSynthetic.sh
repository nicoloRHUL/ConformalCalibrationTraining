#!/bin/bash
python main.py --dataset=synth-linear --modes=fixed-LR-LRExp-sigma-sum
python main.py --dataset=synth-cos --modes=fixed-LR-LRExp-sigma-sum
python main.py --dataset=synth-squared --modes=fixed-LR-LRExp-sigma-sum
python main.py --dataset=synth-inverse --modes=fixed-LR-LRExp-sigma-sum



