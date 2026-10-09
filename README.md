# Event Frequency Analyzer

A local desktop application for finding recurring events in CSV and Excel files.
Python 3.9+; PyQt6 interface, pandas analytics, matplotlib interactive charts, and
SciPy distribution fitting. No server, account, or data upload is required.

## Run

From the repository directory:

```sh
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate
python -m pip install -r requirements.txt
python main.py
```

Linux requires a graphical desktop and the system libraries required by Qt
(including the xcb platform dependencies). For headless GUI checks, set
`QT_QPA_PLATFORM=offscreen`.

## Analyze your files

1. Choose **Add Excel / CSV files** and select one or more files. Nonempty Excel
   sheets are loaded individually, including legacy `.xls` files.
2. Confirm the suggested date and description columns for each source using the
   preview. Different sources can have different schemas. Cancel skips that sheet.
   Enable day-first parsing for ambiguous dates such as `04/05/2024`.
3. Choose **Manual wordset** and enter one search word or phrase per line. Select:
   - **Exact:** whole word/phrase boundaries, not equality with the entire cell.
   - **Partial:** substring matching.
   - **Fuzzy:** approximate matching with a minimum similarity from 0 to 100
     (higher is stricter).
   Matching is case-insensitive unless the checkbox is enabled.
4. Click **Analyze events**. Monthly and trimester charts count each matching
   row once, even if it matches multiple words. Trimesters are calendar quarters
   (January–March, April–June, etc.), not rolling three-month periods.
5. **Top 10 events** groups by your search words/phrases, showing counts, the
   first observed date, and an example description. One row can count towards
   multiple search terms. This is not an automatic semantic classification or
   generated summary in manual mode.
6. **Window comparisons** shows fitted laws and comparisons with the previous
   window and the first-window baseline. Select a row to inspect the histogram
   and candidate curves in **Selected window fit**. Fit parameters and AIC scores
   are also available in the best-fit cell's tooltip.
7. Double-click **Matched entries** to read the full description and optionally
   open the original file. The source sheet and original row number are shown;
   default file viewers do not necessarily support jumping to a specific row.
   The table previews 5,000 entries; **Export matched rows** saves every match.

Dates may mix textual formats, timestamps, and Excel serial dates. Missing or
unparseable dates are reported and excluded. No detector can disambiguate every
numeric identifier or date format: confirm mappings and date ordering. Headers
must occupy the first row; CSV delimiter/encoding detection is best-effort.
Loading the same source sheet twice does not duplicate its rows; clear sources
to reload a changed file. Source files are never modified.

## Plant-construction recognition

Choose **Plant construction** as the analysis mode, then click **Analyze events**.
No wordset is required. The offline recognizer identifies discipline, activity,
issue, and status independently, using a small original English/French starter
dictionary and conservative context rules. It covers civil works, structural
steel, mechanical installation, piping, electrical, instrumentation, and
commissioning; construction activities, delays, missing materials, defects,
rework, nonconformities, access constraints, and common electrical faults.
Synonyms and selected abbreviations map to canonical categories.

For example, cable installation completed, cable installation delayed, and a
negated cable defect are different classifications. Local negation rules avoid
counting a denied defect as a positive fault. Complex or conflicting clauses
are routed to review rather than confidently combining unrelated statuses.

- **Construction classifications** retains every input row with recognized
  fields, rule-based confidence, disposition, evidence, and source references.
- **Needs review / unfamiliar** retains fuzzy suggestions, ambiguous or
  insufficient context, unfamiliar wording, and entries with invalid dates.
- Charts and top-ten summaries include only confidently recognized rows with
  valid dates, counting a row once per chart period. Event labels distinguish
  the canonical context and status, not just a matched synonym.
- **Export all classifications** preserves every row, including review items
  and invalid dates, with the original file, sheet, and row reference. Tables
  preview at most 5,000 rows each; exports are not limited.
- Double-click a classification or review entry to inspect its source.

**Edit construction dictionary** opens a JSON editor for aliases grouped under
`disciplines`, `activities`, `issues`, and `statuses`. **Apply changes** validates
and uses the dictionary for this session; **Save JSON copy** persists it to a
file you choose. Import that copy in future sessions. Invalid schemas and
conflicting aliases within one group are rejected. **Restore starter dictionary**
restores the bundled vocabulary in the editor without modifying your saved file.
The bundled JSON is packaged with the executable; no network access is needed.

**Construction similarity** controls typo suggestions, not semantic similarity.
Short acronyms are not fuzzily guessed. Fuzzy matches require review even when
their spelling score is high. Confidence is a heuristic rule score, not a
calibrated probability. This is not a general language model or an exhaustive
construction ontology: adapt the aliases to your plant, contractors, languages,
and alarm codes, and validate results against representative reports. Unknown
languages and unfamiliar expressions remain visible for review; the application
does not automatically learn from them or approve review entries.

No Electropedia text is scraped or bundled. The vocabulary is an original
starter set, not IEC-certified terminology. Any future imported third-party
dictionary must be reviewed for accuracy and permitted reuse.

## Statistical interpretation

Three-calendar-month windows advance one month at a time. Each window samples
**daily event counts**, including zero-event days, rather than three monthly
totals. Month boundaries are inferred from the date coverage of the source data;
partial boundary months are treated as observed months. Missing days are assumed
to be zero, so incomplete logs may distort results.

Normal, Poisson, Exponential, and Weibull fits are ranked using AIC; unsupported
or degenerate fits are omitted. Continuous laws approximate integer counts using
probabilities over count bins, so their scores can be compared with Poisson.
Continuous parameters are estimated on the raw counts, not optimized for the
binned likelihood, so these AIC rankings are approximate.
The selected law is a heuristic, not a guarantee of goodness of fit.
Comparisons use two-sample tests and a multiple-comparison correction at 5%
significance. Overlapping windows are dependent, and sparse counts, small
samples, or incomplete coverage limit statistical power. Shift flags are
exploratory, not a substitute for domain-specific validation.

## Tests and sample data

Core regression tests use standard-library `unittest`, with a small example
dataset in `tests/sample_events.csv`. No additional test framework is required.

```sh
python -m unittest discover -s tests -v
# On a headless machine:
QT_QPA_PLATFORM=offscreen python -m unittest discover -s tests -v
```

## Build an executable

Build on the target operating system. PyInstaller does not cross-compile:
to create a Windows `.exe`, run these commands on Windows.

```sh
python -m pip install -r requirements-build.txt
python -m PyInstaller --clean --noconfirm event_frequency_analyzer.spec
```

The standalone application is placed in `dist/EventFrequencyAnalyzer.exe`
(without `.exe` on Linux/macOS). Test it on a clean target machine before
distribution. Qt and scientific libraries mean the bundle is larger than a
native minimal viewer; one-file extraction can make initial startup slower.
No Python installation is needed on the destination machine. Review bundled
third-party licenses when distributing the executable.
