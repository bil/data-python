# BIL Data Python API

This is a Python API for accessing and analyzing neural data from the [Brain Interfacing Laboratory](https://bil.stanford.edu/). The public deposition is at the [Stanford Digital Repository (SDR)](https://purl.stanford.edu/zz618yg1930).

## Installation

```bash
pip install bilab
```

## Data Availability

The `bilab` package is an API for interacting with [publicly deposited data](https://purl.stanford.edu/zz618yg1930). In that directory, each recording session has its own identifier under the `data` directory. For instance, "U201130_01" is the first session by subject U on 2020-11-30.

This is a public distribution of an internal API, and **not all data types are publicly available for all sessions.** Public datasets are deposited incrementally. The API contains methods for signals that will raise a `FileNotFoundError` if those signals have not been published. The deposition is growing incrementally to permit reproduction and extension upon published results.

## Quickstart

See `demo/md/quickstart.md`.

```python
from bil.api import get

# Initialize a study using a session ID
# Data will be fetched over HTTPS from the Stanford Digital Repository
study = get("U201130_01", download_dir="my_data")

# Access trial metadata
print(f"Number of trials: {len(study)}")
df = study.df

# Get a span of data for the first trial
# This is an accessor for a continuous segment of time
span = study.spans[0]

# Retrieve LFP data from that period of time from the Utah array in motor cortex
# This well get deposited in my_data/U201130_01
lfp = span.lfp(region="m1")  # (96, T) ndarray
```

## Contributing

This is a public distribution of an internal API; at this time, development is proceeding internally.
