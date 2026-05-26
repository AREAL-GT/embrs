"""Package entry point — `python -m embrs.weather_candidate_search`."""
import sys

from embrs.weather_candidate_search.cli import main

if __name__ == "__main__":
    sys.exit(main())
