# Dataset and attribution

**Approved option 3: YouTube Spam Collection.** One row is one YouTube comment,
with its text, author, timestamp, comment identifier, and spam label (`0 = ham`,
`1 = spam`). The original five CSV files correspond to five music videos.
`dataset.csv` combines them and adds a `video` column derived from the filenames.
Original field values, including `CONTENT`, are preserved during this conversion.

**Citation:** Alberto, T. & Lochter, J. (2015). *YouTube Spam Collection* [Dataset].
UCI Machine Learning Repository. https://doi.org/10.24432/C58885.

- Dataset page: https://archive.ics.uci.edu/dataset/380/youtube+spam+collection
- Exact archive: https://archive.ics.uci.edu/static/public/380/youtube+spam+collection.zip
- License: [Creative Commons Attribution 4.0 International](https://creativecommons.org/licenses/by/4.0/)
- Redistribution is permitted with attribution; the dataset is included in this project.
- Raw size: 1,956 comments across Psy, Katy Perry, LMFAO, Eminem, and Shakira.
- The source advertises no missing values, but the actual CSVs contain 245 blank
  dates. The notebook audits the actual files rather than assuming the metadata is exact.

To recreate the combined CSV, run from the project root:

```bash
python3 scripts/download_data.py
```

This standard-library-only script downloads the official archive, reads the five
expected files in a fixed order, preserves their original columns, and writes
`dataset.csv` with UTF-8 encoding and LF line endings. `SOURCE.json` records the
source URL, archive and combined-CSV SHA-256 hashes, row counts, and citation.
The analysis itself uses the included CSV and needs no network access.
