"""Entry point: ``python -m embrs.scenario_weather <subcommand>``.

A real module entry point with a ``__main__`` guard so FireSim's spotting
workers can re-import safely under multiprocessing/spawn (spec §9).
"""
from embrs.scenario_weather.cli import main

if __name__ == "__main__":
    main()
