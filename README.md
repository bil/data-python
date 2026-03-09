# BIL Data Python API

This repository contains the public-facing Python API for accessing and analyzing neural data from the [Brain Interfacing Laboratory](https://bil.stanford.edu/). THe public deposition is at the [Stanford Digital Repository (SDR)](https://purl.stanford.edu/zz618yg1930).

## Installation

### Local Development

```bash
git clone https://github.com/bil/data-python.git
cd data-python
pip install -e .
```

### GitHub
To install the latest version directly from the repository:

**Via HTTPS:**
```bash
pip install git+https://github.com/bil/data-python.git
```

**Via SSH:**
```bash
pip install git+ssh://git@github.com/bil/data-python.git
```

## Data Availability

The `bil` package is an API for interacting with [publically deposited data](https://purl.stanford.edu/zz618yg1930). In that directory, each recording session has its own identifier under the `data` directory. For instance, "U201130_01" is the first session by subject U on 2020-11-30.

However, **not all data types are available for all sessions.**

Public datasets are deposited incrementally and are often restricted to whatever is required for publication. The API contains methods for signals that will raise a `FileNotFoundError`, if those signals have not been published.

## Quick Start

See `doc/quickstart.ipynb` for more.

```python
from bil.dataclass.dataclassA import Study

# Initialize a study using a session ID
# Data will be fetched over HTTPS from the Stanford Digital Repository
study = Study("U201130_01", download_dir="my_data")

# Access trial metadata
print(f"Number of trials: {len(study)}")
df = study.df

# Get a span of data for the first trial
span = study.spans[0]

# Retrieve LFP data from that trial
lfp = span.lfp(region="m1")
```
