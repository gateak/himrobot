"""Behavior-cloning training for the HIM table-tennis policy.

This package trains a small PyTorch policy that maps robot joint state to
desired joint targets, and saves a self-contained checkpoint that
``him_adapter.policy`` loads at deployment time (falling back to HoldPolicy
when torch or a checkpoint is unavailable).

See training/README.md for provenance and usage.
"""
