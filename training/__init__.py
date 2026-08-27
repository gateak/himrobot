"""Behavior-cloning training for the HIM table-tennis policy.

Trains a small PyTorch policy mapping robot joint state to desired joint
targets, and saves a self-contained checkpoint that ``him_adapter.policy``
loads at deployment (falls back to HoldPolicy if torch or the checkpoint
is unavailable). See training/README.md for provenance and usage.
"""
