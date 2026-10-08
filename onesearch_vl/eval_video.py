#!/usr/bin/env python3
"""Evaluate OneSearch-Video or VideoDR trajectories."""
import sys

from eval_common import run_evaluation


def main(argv=None, judge_fn=None) -> int:
    return run_evaluation(argv, judge_fn, modality='video')


if __name__ == '__main__':
    sys.exit(main())
