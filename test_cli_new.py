#!/usr/bin/env python3
"""Test CLI parsing for new flags."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from config.config_manager import parse_cli_args

if __name__ == "__main__":
    args = parse_cli_args()
    print(f"Parsed CLI args successfully!")
    print(f"use_base_model_only: {getattr(args, 'use_base_model_only', 'NOT FOUND')}")
    print(f"no_use_agent_chain: {getattr(args, 'no_use_agent_chain', 'NOT FOUND')}")
