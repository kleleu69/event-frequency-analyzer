"""Load spreadsheets without silently discarding data rows."""

import csv
import io
import numbers
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List

import pandas as pd

from src.utils.dates import parse_dates


class FileLoadError(ValueError):
    """A readable error suitable for display in a file-import dialog."""


@dataclass
class LoadedSheet:
    path: str
    sheet: str
    frame: pd.DataFrame
    date_candidates: List[str]
    text_candidates: List[str]


def _candidates(frame):
    dates, texts = [], []
    for column in frame.columns:
        values = frame[column].dropna()
        if values.empty:
            continue
        sample = values.head(100)
        date_header = bool(re.search(r"date|time|timestamp", column, re.I))
        numeric = all(isinstance(value, numbers.Number) for value in sample)
        ratio = parse_dates(sample).notna().mean()
        if ratio >= 0.6 and (date_header or not numeric):
            dates.append(column)
        if any(isinstance(value, str) for value in sample):
            texts.append(column)
    return dates, texts


class FileLoader:
    @staticmethod
    def load(path) -> List[LoadedSheet]:
        """Load CSV or every nonempty Excel sheet; headers occupy row one."""
        file_path = Path(path)
        suffix = file_path.suffix.lower()
        if suffix not in (".csv", ".xlsx", ".xls"):
            raise FileLoadError("Unsupported file type. Choose CSV, XLSX, or XLS.")
        try:
            if suffix == ".csv":
                raw = file_path.read_bytes()
                if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
                    encodings = ["utf-16"]
                else:
                    encodings = ["utf-8-sig", "cp1252", "latin-1"]
                text = None
                for encoding in encodings:
                    try:
                        text = raw.decode(encoding)
                        break
                    except UnicodeDecodeError:
                        continue
                try:
                    delimiter = csv.Sniffer().sniff(text[:65536],
                                                   delimiters=",;\t|").delimiter
                except csv.Error:
                    delimiter = ","
                sheets = {"CSV": pd.read_csv(io.StringIO(text), sep=delimiter,
                                             skip_blank_lines=False)}
            else:
                sheets = pd.read_excel(file_path, sheet_name=None,
                                       engine="xlrd" if suffix == ".xls" else "openpyxl")
            result = []
            for name, frame in sheets.items():
                if frame.empty or frame.dropna(how="all").empty:
                    continue
                frame.columns = [str(column) for column in frame.columns]
                date_candidates, text_candidates = _candidates(frame)
                result.append(LoadedSheet(str(file_path), str(name), frame,
                                          date_candidates, text_candidates))
            return result
        except FileNotFoundError:
            raise FileLoadError("File not found: {}".format(file_path)) from None
        except PermissionError:
            raise FileLoadError("Cannot read the selected file: permission denied.") from None
        except ImportError:
            raise FileLoadError("The Excel reader dependency is unavailable.") from None
        except Exception as error:
            raise FileLoadError("Could not load {}: {}".format(file_path.name, error)) from None
