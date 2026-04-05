# BIL Data Python API

Welcome! This notebook introduces the `bil` data API for accessing
publicly released data from the [Brain Interfacing
Lab](https://bil.stanford.edu/).

This package lets you interact with data publicly deposited to the
[Stanford Digital Repository
(SDR)](https://purl.stanford.edu/zz618yg1930). It uses the `requests`
package to download files over HTTPS upon access.

First, install the package!

**Via HTTPS:**

```bash
pip install git+https://github.com/bil/data-python.git
```

**Via SSH:**

```bash
pip install git+ssh://git@github.com/bil/data-python.git
```

## Data Availability Disclaimer

1. **Incremental Releases:** The public deposition in the
   [SDR](https://purl.stanford.edu/zz618yg1930) is being expanded
   incrementally and only contains data associated with formal
   publications.
1. **Missing Files:** Because of this, many API methods **will not
   work** until the corresponding data has been released. These will
   fail with a `FileNotFoundError`.

## The Study

The simplest entry point is the `get` method. If you are seeking data
from a session ID in a paper (it will look something like “U201130_01”,
meaning the first session by subject “U” on 2020-11-30), you can
initialize a `Study` object pointing to that session.

The initialization will automatically choose the correct format. Below,
you’ll see that we end up with an object from `formatS`.

The full set of constructor arguments is: - `study_id`: The session
identifier (e.g., “U201130_01”). - `download_dir`: The local directory
where data will be stored. By default, this is a relative path:
“data”. - `quiet`: A flag to suppress progress bars and print statements
during data transfer (defaults to `False`).

Useful attributes of the resultant object include: - `url`: The remote
HTTPS endpoint from which data will be fetched. This is set from a
constant. - `has`: A set of data (either a semantic name or a path
string) that has been downloaded by this object. It will be empty at
first.

Instantiation prepares the `download_dir` but **does not download
anything yet**.

```python
import os
import numpy as np
from bil.api import get

study = get("U201130_01", download_dir="tmp")
print(f"Study came from class: {study.__class__}\n")

print(f"Study ID: {study.study_id}")
print(f"Local Cache Directory: {study.download_dir}")
print(f"Remote URL: {study.url}")
print(f"Set of downloaded data: {study.has}")
```

```
Copying U201130_01.yaml to tmp/U201130_01...


U201130_01.yaml: 0.00iB [00:00, ?iB/s]
U201130_01.yaml: 48.0iB [00:00, 37.1kiB/s]

Study came from class: <class 'bil.api.formatS.Study'>

Study ID: U201130_01
Local Cache Directory: tmp/U201130_01
Remote URL: https://stacks.stanford.edu/file/zz618yg1930/data/U201130_01
Set of downloaded data: set()
```

## Accessing Metadata

For some sessions, data is organized into trials. This example is one of
those. Specifically, this dataset features subject U performing radial,
center-out reaches to 8 targets.

The number of trials defines the `Study`’s length. Each trial has
associated metadata, which is fetched from the server when you access
attributes such as the length or the `df` property.

```python
# Fetch trial metadata and create Span objects
num_trials = len(study)
print(f"Total number of trials: {num_trials}")

# The trial metadata is available as a pandas DataFrame
display(study.df.head())
```

```
Copying df/trial.df.xz to tmp/U201130_01...


df/trial.df.xz: 0.00iB [00:00, ?iB/s]
df/trial.df.xz: 25.6kiB [00:00, 7.03MiB/s]

Total number of trials: 1500
```

<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }
&#10;    .dataframe tbody tr th {
        vertical-align: top;
    }
&#10;    .dataframe thead th {
        text-align: right;
    }
</style>

|     | study      | number | trial_id     | ms_start | ms_end | success | quality | target_loc_x | target_loc_y | target_loc | ... | pe_cursor_offset_y | pe_task_toggle | pe_juice_toggle | pe_time_max_last_success | pe_cursor_offset_x_toggle | pe_z_window_toggle | pe_target_color_value | pe_task_automation | pe_target_stack_on | pe_radius_target |
| --- | ---------- | ------ | ------------ | -------- | ------ | ------- | ------- | ------------ | ------------ | ---------- | --- | ------------------ | -------------- | --------------- | ------------------------ | ------------------------- | ------------------ | --------------------- | ------------------ | ------------------ | ---------------- |
| 0   | U201130_01 | 0      | U201130_01_0 | 759      | 1330   | True    | 1.0     | 0            | 0            | center     | ... | -1                 | -1             | -1              | -1                       | -1                        | -1                 | -1                    | -1                 | -1                 | -1               |
| 1   | U201130_01 | 1      | U201130_01_1 | 1345     | 2013   | True    | 1.0     | 100          | 0            | right      | ... | -1                 | -1             | -1              | -1                       | -1                        | -1                 | -1                    | -1                 | -1                 | -1               |
| 2   | U201130_01 | 2      | U201130_01_2 | 2028     | 2680   | True    | 1.0     | 0            | 0            | center     | ... | -1                 | -1             | -1              | -1                       | -1                        | -1                 | -1                    | -1                 | -1                 | -1               |
| 3   | U201130_01 | 3      | U201130_01_3 | 2695     | 3446   | True    | 1.0     | 0            | -100         | bottom     | ... | -1                 | -1             | -1              | -1                       | -1                        | -1                 | -1                    | -1                 | -1                 | -1               |
| 4   | U201130_01 | 4      | U201130_01_4 | 3461     | 4263   | True    | 1.0     | 0            | 0            | center     | ... | -1                 | -1             | -1              | -1                       | -1                        | -1                 | -1                    | -1                 | -1                 | -1               |

<p>5 rows × 73 columns</p>
</div>

The columns of this DataFrame can be accessed by string indexing:

```python
print(f"Target locations: {study['target_loc'].unique()}")
```

```
Target locations: ['center' 'right' 'bottom' 'bottom_right' 'top_left' 'bottom_left' 'top'
 'top_right' 'left']
```

## Spans

A `Study` object is a wrapper used to retrieve `Span` objects. A `Span`
is an abstraction for any continuous segment of time, using units of
milliseconds. Most of our data signals are sampled or binned at 1kHz, so
accessing a `Span`’s data returns a timepoint for each millisecond. This
also means the smallest `Span` you can have is 1 millisecond long (the
largest is the length of the session).

Because this session has a trial structure, it already holds a list of
`Span` objects corresponding to each trial duration. These are
accessible via `study.spans`. These are iterated over when iterating
through the `Study` and can be accessed via integer indexing.

```python
# Get the span for the very first trial
first_trial_span = study.spans[0]

print(f"Trial Start: {first_trial_span.start} ms")
print(f"Trial End: {first_trial_span.stop} ms")
```

```
Trial Start: 759 ms
Trial End: 1330 ms
```

The length of a `Span` is its duration in milliseconds:

```python
print(len(first_trial_span))
print(first_trial_span.stop - first_trial_span.start)
```

```
571
571
```

A `Span` has a `metadata` attribute:

```python
print(first_trial_span.metadata["target_loc"])
```

```
center
```

Another way to get a span is to index by time. Specifically, you can get
the first 100ms of the whole session by:

```python
print(study.by_time(slice(None, 100)))
```

```
Copying U201130_01.h5 to tmp/U201130_01...


U201130_01.h5: 0.00iB [00:00, ?iB/s]
U201130_01.h5: 221kiB [00:00, 15.4MiB/s]

span(study_id=U201130_01, start=0ms, stop=100ms)
```

Or the whole session by:

```python
print(study.by_time(slice(None, None)))
```

```
span(study_id=U201130_01, start=0ms, stop=1359000ms)
```

## Accessing Data

Now let’s retrieve some actual neural and behavioral data for our trial!

When you call these methods, the object checks if the requested file is
cached locally. If not, it will be downloaded automatically. Once
downloaded, it returns data from the requested span of time.

```python
import matplotlib.pyplot as plt

# 1. Fetch Cursor Kinematics (2D position)
kinematics = first_trial_span.kinematics()
print("Kinematics shape (dimensions, time):", kinematics.shape)

# 2. Fetch Local Field Potential (LFP) data for a specific brain region ('m1')
# Note: This will download the lfp_1k.h5 file the first time!
lfp = first_trial_span.lfp(region="m1")
print("LFP shape (channels, time):", lfp.shape)

# Plot the data
fig, axs = plt.subplots(2, 1, figsize=(6, 6), sharex=True, layout="constrained")

# Plot cursor position
axs[0].plot(kinematics[0, :], label="X-Position")
axs[0].plot(kinematics[1, :], label="Y-Position")
axs[0].set_ylabel("Position (mm)")
axs[0].legend()
axs[0].set_title("Behavioral Kinematics")

# Plot LFP
axs[1].imshow(lfp, aspect="auto", cmap="viridis", origin="lower")
axs[1].set_ylabel("Channel")
axs[1].set_xlabel("Time (ms)")
axs[1].set_title("Local Field Potential")
plt.show()
```

```
Copying signal/pos_cursor_x.h5 to tmp/U201130_01...


signal/pos_cursor_x.h5: 0.00iB [00:00, ?iB/s]
signal/pos_cursor_x.h5: 8.21MiB [00:00, 82.1MiB/s]
signal/pos_cursor_x.h5: 10.9MiB [00:00, 28.8MiB/s]

Copying signal/pos_cursor_y.h5 to tmp/U201130_01...


signal/pos_cursor_y.h5: 0.00iB [00:00, ?iB/s]
signal/pos_cursor_y.h5: 10.9MiB [00:00, 30.8MiB/s]

Kinematics shape (dimensions, time): (2, 571)
Copying neural/lfp_1k.h5 to tmp/U201130_01...


neural/lfp_1k.h5: 0.00iB [00:00, ?iB/s]
neural/lfp_1k.h5: 11.4MiB [00:00, 114MiB/s]
neural/lfp_1k.h5: 22.8MiB [00:00, 100MiB/s]
neural/lfp_1k.h5: 32.9MiB [00:00, 93.9MiB/s]
neural/lfp_1k.h5: 42.7MiB [00:00, 95.3MiB/s]
neural/lfp_1k.h5: 52.3MiB [00:00, 90.8MiB/s]
neural/lfp_1k.h5: 61.7MiB [00:00, 92.0MiB/s]
neural/lfp_1k.h5: 71.6MiB [00:00, 94.1MiB/s]
neural/lfp_1k.h5: 81.1MiB [00:00, 91.7MiB/s]
neural/lfp_1k.h5: 91.0MiB [00:00, 93.9MiB/s]
neural/lfp_1k.h5: 100MiB [00:01, 83.3MiB/s]
neural/lfp_1k.h5: 109MiB [00:01, 74.7MiB/s]
neural/lfp_1k.h5: 117MiB [00:01, 73.1MiB/s]
neural/lfp_1k.h5: 124MiB [00:01, 69.0MiB/s]
neural/lfp_1k.h5: 131MiB [00:01, 67.4MiB/s]
neural/lfp_1k.h5: 138MiB [00:01, 65.2MiB/s]
neural/lfp_1k.h5: 145MiB [00:01, 63.0MiB/s]
neural/lfp_1k.h5: 151MiB [00:01, 61.4MiB/s]
neural/lfp_1k.h5: 157MiB [00:02, 61.7MiB/s]
neural/lfp_1k.h5: 164MiB [00:02, 63.2MiB/s]
neural/lfp_1k.h5: 170MiB [00:02, 62.4MiB/s]
neural/lfp_1k.h5: 177MiB [00:02, 63.4MiB/s]
neural/lfp_1k.h5: 184MiB [00:02, 65.7MiB/s]
neural/lfp_1k.h5: 191MiB [00:02, 66.3MiB/s]
neural/lfp_1k.h5: 197MiB [00:02, 65.8MiB/s]
neural/lfp_1k.h5: 204MiB [00:02, 65.6MiB/s]
neural/lfp_1k.h5: 211MiB [00:02, 66.2MiB/s]
neural/lfp_1k.h5: 217MiB [00:02, 65.2MiB/s]
neural/lfp_1k.h5: 224MiB [00:03, 65.3MiB/s]
neural/lfp_1k.h5: 231MiB [00:03, 60.5MiB/s]
neural/lfp_1k.h5: 237MiB [00:03, 59.0MiB/s]
neural/lfp_1k.h5: 243MiB [00:03, 61.1MiB/s]
neural/lfp_1k.h5: 251MiB [00:03, 65.4MiB/s]
neural/lfp_1k.h5: 258MiB [00:03, 63.9MiB/s]
neural/lfp_1k.h5: 264MiB [00:03, 65.3MiB/s]
neural/lfp_1k.h5: 271MiB [00:03, 66.0MiB/s]
neural/lfp_1k.h5: 278MiB [00:03, 62.6MiB/s]
neural/lfp_1k.h5: 284MiB [00:04, 63.2MiB/s]
neural/lfp_1k.h5: 291MiB [00:04, 60.0MiB/s]
neural/lfp_1k.h5: 297MiB [00:04, 57.8MiB/s]
neural/lfp_1k.h5: 303MiB [00:04, 60.0MiB/s]
neural/lfp_1k.h5: 309MiB [00:04, 59.7MiB/s]
neural/lfp_1k.h5: 316MiB [00:04, 60.9MiB/s]
neural/lfp_1k.h5: 323MiB [00:04, 62.5MiB/s]
neural/lfp_1k.h5: 329MiB [00:04, 61.1MiB/s]
neural/lfp_1k.h5: 335MiB [00:04, 60.5MiB/s]
neural/lfp_1k.h5: 341MiB [00:04, 59.3MiB/s]
neural/lfp_1k.h5: 348MiB [00:05, 63.2MiB/s]
neural/lfp_1k.h5: 355MiB [00:05, 59.3MiB/s]
neural/lfp_1k.h5: 362MiB [00:05, 62.0MiB/s]
neural/lfp_1k.h5: 368MiB [00:05, 63.7MiB/s]
neural/lfp_1k.h5: 376MiB [00:05, 66.8MiB/s]
neural/lfp_1k.h5: 384MiB [00:05, 70.9MiB/s]
neural/lfp_1k.h5: 391MiB [00:05, 67.3MiB/s]
neural/lfp_1k.h5: 398MiB [00:05, 68.0MiB/s]
neural/lfp_1k.h5: 405MiB [00:05, 67.7MiB/s]
neural/lfp_1k.h5: 412MiB [00:06, 69.4MiB/s]
neural/lfp_1k.h5: 419MiB [00:06, 69.1MiB/s]
neural/lfp_1k.h5: 426MiB [00:06, 65.3MiB/s]
neural/lfp_1k.h5: 433MiB [00:06, 63.6MiB/s]
neural/lfp_1k.h5: 440MiB [00:06, 66.2MiB/s]
neural/lfp_1k.h5: 447MiB [00:06, 66.7MiB/s]
neural/lfp_1k.h5: 454MiB [00:06, 64.3MiB/s]
neural/lfp_1k.h5: 461MiB [00:06, 67.3MiB/s]
neural/lfp_1k.h5: 468MiB [00:06, 66.5MiB/s]
neural/lfp_1k.h5: 475MiB [00:07, 61.3MiB/s]
neural/lfp_1k.h5: 482MiB [00:07, 63.7MiB/s]
neural/lfp_1k.h5: 488MiB [00:07, 58.5MiB/s]
neural/lfp_1k.h5: 495MiB [00:07, 62.1MiB/s]
neural/lfp_1k.h5: 502MiB [00:07, 64.6MiB/s]
neural/lfp_1k.h5: 509MiB [00:07, 65.3MiB/s]
neural/lfp_1k.h5: 516MiB [00:07, 62.8MiB/s]
neural/lfp_1k.h5: 522MiB [00:07, 62.6MiB/s]
neural/lfp_1k.h5: 530MiB [00:07, 66.7MiB/s]
neural/lfp_1k.h5: 536MiB [00:07, 65.8MiB/s]
neural/lfp_1k.h5: 544MiB [00:08, 67.7MiB/s]
neural/lfp_1k.h5: 552MiB [00:08, 71.3MiB/s]
neural/lfp_1k.h5: 559MiB [00:08, 68.3MiB/s]
neural/lfp_1k.h5: 566MiB [00:08, 60.5MiB/s]
neural/lfp_1k.h5: 572MiB [00:08, 60.4MiB/s]
neural/lfp_1k.h5: 579MiB [00:08, 61.7MiB/s]
neural/lfp_1k.h5: 585MiB [00:08, 60.6MiB/s]
neural/lfp_1k.h5: 591MiB [00:08, 61.3MiB/s]
neural/lfp_1k.h5: 598MiB [00:08, 63.3MiB/s]
neural/lfp_1k.h5: 605MiB [00:09, 63.8MiB/s]
neural/lfp_1k.h5: 611MiB [00:09, 63.6MiB/s]
neural/lfp_1k.h5: 617MiB [00:09, 61.0MiB/s]
neural/lfp_1k.h5: 624MiB [00:09, 60.9MiB/s]
neural/lfp_1k.h5: 630MiB [00:09, 62.5MiB/s]
neural/lfp_1k.h5: 637MiB [00:09, 64.1MiB/s]
neural/lfp_1k.h5: 643MiB [00:09, 63.2MiB/s]
neural/lfp_1k.h5: 651MiB [00:09, 66.5MiB/s]
neural/lfp_1k.h5: 658MiB [00:09, 64.6MiB/s]
neural/lfp_1k.h5: 664MiB [00:09, 65.4MiB/s]
neural/lfp_1k.h5: 671MiB [00:10, 64.8MiB/s]
neural/lfp_1k.h5: 677MiB [00:10, 63.2MiB/s]
neural/lfp_1k.h5: 685MiB [00:10, 66.1MiB/s]
neural/lfp_1k.h5: 691MiB [00:10, 64.8MiB/s]
neural/lfp_1k.h5: 698MiB [00:10, 47.9MiB/s]
neural/lfp_1k.h5: 703MiB [00:10, 44.2MiB/s]
neural/lfp_1k.h5: 708MiB [00:10, 44.1MiB/s]
neural/lfp_1k.h5: 713MiB [00:10, 45.4MiB/s]
neural/lfp_1k.h5: 720MiB [00:11, 50.3MiB/s]
neural/lfp_1k.h5: 725MiB [00:11, 49.0MiB/s]
neural/lfp_1k.h5: 733MiB [00:11, 57.0MiB/s]
neural/lfp_1k.h5: 740MiB [00:11, 62.4MiB/s]
neural/lfp_1k.h5: 749MiB [00:11, 70.2MiB/s]
neural/lfp_1k.h5: 758MiB [00:11, 75.1MiB/s]
neural/lfp_1k.h5: 767MiB [00:11, 78.1MiB/s]
neural/lfp_1k.h5: 775MiB [00:11, 67.6MiB/s]
neural/lfp_1k.h5: 783MiB [00:11, 72.0MiB/s]
neural/lfp_1k.h5: 790MiB [00:12, 71.0MiB/s]
neural/lfp_1k.h5: 798MiB [00:12, 67.1MiB/s]
neural/lfp_1k.h5: 805MiB [00:12, 69.2MiB/s]
neural/lfp_1k.h5: 812MiB [00:12, 67.0MiB/s]
neural/lfp_1k.h5: 819MiB [00:12, 65.1MiB/s]
neural/lfp_1k.h5: 826MiB [00:12, 67.1MiB/s]
neural/lfp_1k.h5: 833MiB [00:12, 64.5MiB/s]
neural/lfp_1k.h5: 840MiB [00:12, 61.1MiB/s]
neural/lfp_1k.h5: 846MiB [00:12, 60.0MiB/s]
neural/lfp_1k.h5: 852MiB [00:13, 60.7MiB/s]
neural/lfp_1k.h5: 858MiB [00:13, 51.5MiB/s]
neural/lfp_1k.h5: 866MiB [00:13, 57.0MiB/s]
neural/lfp_1k.h5: 873MiB [00:13, 61.2MiB/s]
neural/lfp_1k.h5: 879MiB [00:13, 58.0MiB/s]
neural/lfp_1k.h5: 885MiB [00:13, 56.2MiB/s]
neural/lfp_1k.h5: 892MiB [00:13, 58.0MiB/s]
neural/lfp_1k.h5: 897MiB [00:13, 58.0MiB/s]
neural/lfp_1k.h5: 903MiB [00:13, 57.1MiB/s]
neural/lfp_1k.h5: 905MiB [00:28, 57.1MiB/s]
neural/lfp_1k.h5: 905MiB [00:35, 25.3MiB/s]

LFP shape (channels, time): (96, 571)
```

![](quickstart_files/figure-commonmark/cell-10-output-8.png)

To see all built-in data access functions, you can investigate the
`DataCatalog` implementation in `bil.api.formatS`. As mentioned earlier,
many of those functions may not work if the corresponding data hasn’t
been deposited. For instance, the large raw data files will often not
exist:

```python
try:
    raw = first_trial_span.raw(region="m1")
except FileNotFoundError as e:
    print(f"Raw data doesn't exist for this session!\n\n{e}")
```

```
Copying neural/raw_ch001.h5 to tmp/U201130_01...
Raw data doesn't exist for this session!

Failed to download https://stacks.stanford.edu/file/zz618yg1930/data/U201130_01/neural/raw_ch001.h5: 404 Client Error: Not Found for url: https://stacks.stanford.edu/file/zz618yg1930/data/U201130_01/neural/raw_ch001.h5
```

## SpanSet and SpanArray

Just as integer indexing returns a trial `Span`, slice indexing returns
a set of spans, or a `SpanSet`. This is a wrapper for a list of `Span`
objects that returns data from each span, collected into a list.

```python
spanset = study[:5]
print(f"Type: {type(spanset)}")
print(f"Length: {len(spanset)}")
```

```
Type: <class 'bil.api.formatS.SpanSet'>
Length: 5
```

We can get the kinematics from these trials:

```python
list_of_kinematics = spanset.kinematics()
for kinematics in list_of_kinematics:
    print(kinematics.shape)
```

```
(2, 571)
(2, 668)
(2, 652)
(2, 751)
(2, 802)
```

Since each `Span` can have a different length, these methods return
lists of differently sized arrays. However, a `SpanArray` is a version
of a `SpanSet` where each underlying `Span` has the same length! The
easiest way to create one is with the `around` method. Here is an
example of getting the 500 milliseconds surrounding the time the
reaching target was acquired, using the DataFrame key
`time_target_acquire_last`.

```python
spanarray = spanset.around("time_target_acquire_last", t_before=250, t_after=249)
array_of_kinematics = spanarray.kinematics()
print(f"Shape: {array_of_kinematics.shape}")
```

```
Shape: (5, 2, 500)
```
