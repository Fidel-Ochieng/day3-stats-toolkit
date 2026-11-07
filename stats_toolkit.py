"""
Day 3 - Statistical Analysis Toolkit
Squad Python Specialization - Data Science, AI & Analytics

RULE: implement every calculation yourself with plain Python.
NumPy may be imported ONLY in compare_with_numpy(), to verify your results.
Do not use the statistics module.
"""
import csv
import math
from pathlib import Path

DATA_PATH = Path("data") / "agent_deposits.csv"


def rejection_reason(raw: str) -> str | None:
    """Return why this raw text is not a valid deposit, or None if it is valid."""
    text = (raw or "").strip()
    if text == "":
        return "missing value"
    if text.lower() in ("-", "--", "n/a", "na", "null" "none"):
        return "placeholder, not a number"
    try:
        number = float(text)
    except ValueError:
        return "not a valid number"
    if not math.isfinite(number):
        return "not a finite number"
    if number < 0:
        return "negative value"
    return None


def load_column(path: Path, column: str) -> tuple[list[float], list[tuple[int, str]]]:
    """Return (valid_values, rejected) where rejected holds (row_number, raw_value) pairs."""

    valid_values = []
    rejected = []
    with path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None or column not in reader.fieldnames:
            raise ValueError(f"column '{column}' not found in {path.name}")
        for row_number, row in enumerate(reader, start=1):
            raw = row[column]
            if rejection_reason(raw) is not None:
                rejected.append((row_number, raw))
                continue
            valid_values.append(float(raw))
    return valid_values, rejected




def mean(values: list[float]) -> float:
    if len(values) == 0:
        raise ValueError("mean() needs at least one value")
    return sum(values) / len(values)



def median(values: list[float]) -> float:
    if len(values) == 0:
        raise ValueError("median() needs at least one value")
    ordered = sorted(values)
    count = len(ordered)
    middle = count // 2
    if count % 2 == 1:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2


def mode(values: list[float]) -> list[float]:
    """Return every value that occurs most often (a list, because there can be ties or none)."""
    if len(values) == 0:
        raise ValueError("mode() needs at least one value")
    counts = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    highest = max(counts.values())
    if highest == 1:
        return []
    return sorted(value for value, count in counts.items() if count == highest)


def variance(values: list[float], sample: bool = True) -> float:
    """Sample variance (n - 1) by default; population variance (n) when sample is False."""
    count = len(values)
    if count == 0:
        raise ValueError("variance() needs at least one value")
    if sample and count < 2:
        raise ValueError("sample variance needs at least two values")
    average = sum(values) / count
    squared_distances = sum((value - average) ** 2 for value in values)
    if sample:
        return squared_distances / (count - 1)

    return squared_distances / count


def std_dev(values: list[float], sample: bool = True) -> float:
    return math.sqrt(variance(values, sample))

def _percentile(ordered: list[float], fraction: float) -> float:
    """Vsalue at this fraction of the way up a sorted list (linear interpolation)."""
    position = (len(ordered)-1)* fraction
    below = math.floor(position)
    above = math.ceil(position)
    return ordered[below] + (ordered[above] - ordered[below]) * (position - below)

def quartiles(values: list[float]) -> tuple[float, float, float]:
    """Return (Q1, Q2, Q3). Document which quartile method you chose."""
    if len(values) == 0:
        raise ValueError("quartiles() needs at least one value")
    ordered = sorted(values)
    return (

        _percentile(ordered, 0.25),
        _percentile(ordered, 0.50),
        _percentile(ordered, 0.75),
    )

def iqr_outliers(values: list[float], k: float = 1.5) -> dict:
    """Return the IQR, the lower and upper fences, and the values outside them."""
    if len(values) == 0:
        raise ValueError("iqr_outliers() needs at least one value")
    if k < 0:
        raise ValueError("k must not be negative")
    q1, _q2, q3 = quartiles(values)
    iqr = q3 - q1
    lower = q1 - k * iqr
    upper = q3 + k * iqr
    outliers = [value for value in values if value < lower or value > upper]
    return {
        "q1": q1,
        "q3": q3,
        "iqr": iqr,
        "lower_fence": lower,
        "upper_fence": upper,
        "outliers": outliers,
    }


def money(number):
    return "n/a" if number is None else f"{number:,.2f}"


class StatisticsAnalyzer:
    """Wraps one numeric column and exposes every statistic above as a method."""

    def __init__(self, values: list[float], name: str = "values") -> None:
        if len(values) == 0:
            raise ValueError("StatisticsAnalyzer needs at least one valid value")
        self.name = name
        self.values = list(values)
        self.sorted_values = sorted(values)
        self.rejected = []
        self.rows_read = len(values)
        self.source_path = None
        self.reasons = {}
        self.labels = {}
        self._summary = None


    @classmethod
    def from_csv(cls, path: Path, column: str) -> "StatisticsAnalyzer":
        values, rejected = load_column(path, column)
        analyzer = cls(values, name=column)
        analyzer.rejected = rejected
        analyzer.reasons = {row_number: rejection_reason(raw) for row_number, raw in rejected}
        with path.open("r", newline="",encoding="utf-8") as file:
            for row in csv.DictReader(file):
                if rejection_reason(row[column]) is None:
                    analyzer.labels[float(row[column])] = "(" + row["date"] + ", " + row["agent_id"] + ")"
        analyzer.rows_read = len(values) + len(rejected)
        analyzer.source_path = path
        return analyzer

    def summary(self) -> dict:
        if self._summary is not None:
            return self._summary
        data = self.sorted_values
        q1 = _percentile(data, 0.25)
        q3 = _percentile(data, 0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        enough = len(data) >= 2
        self._summmary = {
            "count": len(data),
            "mean": mean(data),
            "median": _percentile(data, 0.5),
            "mode": mode(data),
            "variance": variance(data) if enough else None,
            "std_dev": std_dev(data) if enough else None,
            "min": data[0],
            "max": data[-1],
            "q1": q1,
            "q3": q3,
            "iqr": iqr,
            "lower_fence": lower,
            "upper_fence": upper,
            "outliers": [value for value in self.values if value < lower or value > upper],

        }
        return self._summmary

    def without_outliers(self) -> "StatisticsAnalyzer":
        s = self.summary()
        kept = [v for v in self.values if s["lower_fence"] <= v <= s["upper_fence"]]
        cleaned = StatisticsAnalyzer(kept, name=self.name + " (out;oers removed)")
        cleaned.labels = self.labels
        return cleaned


    def report(self) -> str:
        a = self.summary()
        b = self.without_outliers().summary()

        lines = []
        lines.append(f"=== Statistics report: {self.name } ===")
        lines.append(f"Rows read             : {self.rows_read}")
        lines.append(f"Valid Values          : {a['count']}")
        lines.append(f"Rejected values       : {len(self.rejected)}")
        for row_number, raw in self.rejected:
            reason = self.reasons.get(row_number, "not a valid number")
            lines.append(f"  row {row_number}: '{raw}' ({reason})")
        lines.append("")
        lines.append(f"{'':<20}{'all data' :<26}outliers removed")
        rows = [
                    ("count", str(a["count"]), str(b["count"])),
                    ("mean", money(a["mean"]), money(b["mean"])),
                    ("median", money(a["median"]), money(b["median"])),
                    (
                        "mode",
                          ", ".join(money(m) for m in a["mode"]) or "none",
                        ", ".join(money(m) for m in b["mode"]) or "none",
                    ),
                    ("variance (sample)", money(a["variance"]), money(b["variance"])),
                    ("std dev (sample)", money(a["std_dev"]), money(b["std_dev"])),
                    (
                        "min / max",
                        f"{money(a['min'])} / {money(a['max'])}",
                        f"{money(b['min'])} / {money(b['max'])}",
                    ),
                    (
                        "Q1 / Q3",
                        f"{money(a['q1'])} / {money(a['q3'])}",
                        f"{money(b['q1'])} / {money(b['q3'])}",
                    ),
                    ("IQR", money(a["iqr"]), money(b["iqr"])),
                ]
        for label, left, right in rows:
            lines.append(f"{label:<20}{left:<26}{right}")

        found = {}
        with self.source_path.open("r", newline="", encoding="utf-8") as file:
            for row in csv.DictReader(file):
                try:
                    found[float(row["daily_deposits_kes"])] = "(" + row["date"] + ", " + row["agent_id"] + ")"
                except ValueError:
                    pass

        outlier_text = []
        for value in a["outliers"]:
            label = self.labels.get(value)
            outlier_text.append(money(value) + (" " + label if label else ""))

        lines.append("")
        lines.append(f"Outlier fences     :lower {money(a['lower_fence'])}, upper {money(a['upper_fence'])}")
        lines.append("Outliers       :" + (", ".join(outlier_text) or "none"))

        return "\n".join(lines)


def compare_with_numpy(analyzer: StatisticsAnalyzer) -> dict[str, bool]:
    """Import numpy HERE, recompute each statistic and return {name: matches}."""
    import numpy as np
    data = np.array(analyzer.values)
    s = analyzer.summary()
    q1, q3 = np.percentile(data, [25, 75], method="linear")
    return {
        "mean": math.isclose(s["mean"], np.mean(data)),
        "median": math.isclose(s["median"], np.median(data)),
        "variance": math.isclose(s["variance"], np.var(data, ddof=1)),
        "std dev": math.isclose(s["std_dev"], np.std(data, ddof=1)),
        "Q1 / Q3": math.isclose(s["q1"], q1) and math.isclose(s["q3"], q3),

    }


if __name__ == "__main__":
    import subprocess
    import sys
    values, rejected = load_column(DATA_PATH, "daily_deposits_kes")
    print("valid:", len(values))
    print("Rejected:", rejected)

    print("mean:", mean([2, 4, 6]))
    print("median odd:", median([3, 1, 2]))
    print("median even:", median([4, 1, 3, 2]))
    print("mode with a tie:", mode([1, 1, 2, 2, 3]))
    print("mode with no repeats:", mode([1, 2, 3]))

    sample_data = [2, 4, 4, 4, 5, 5, 7, 9]
    print("population variance:", variance(sample_data, sample=False))
    print("sample variance:", variance(sample_data))
    print("variance of data:", variance(values))

    print("population std dev:", std_dev(sample_data, sample=False))
    print("sample std dev:", std_dev(sample_data))
    print("std dev of data:", std_dev(values))

    print("quartiles [1..5]:", quartiles([1, 2, 3, 4, 5]))
    print("quartiles [4,3,2,1]:", quartiles([4, 3, 2, 1]))
    print("quartiles of data:", quartiles(values))

    print("small outlier test:",iqr_outliers([10, 11, 12, 13, 14, 15, 16, 100]))
    result = iqr_outliers(values)
    print("IQR:", result["iqr"])
    print("fence:", result["lower_fence"], result["upper_fence"])
    print("outliers:", result["outliers"])

    analyzer = StatisticsAnalyzer.from_csv(DATA_PATH, "daily_deposits_kes")
    s = analyzer.summary()
    print("count:", s["count"], "mean:", s["mean"], "median:", s["median"])
    cleaned = analyzer.without_outliers()
    print("cleaned count:", cleaned.summary()["count"])
    print("original count still:", analyzer.summary()["count"])
    results = compare_with_numpy(analyzer)
    test_run = subprocess.run([sys.executable, "-m", "unittest"], capture_output=True, text=True)
    ran = next((line for line in test_run.stderr.splitlines() if line.startswith("Ran ")), "tests did not run")


    extra = ["", "NumPy verification"]
    for name, matches in results.items():
        extra.append(f" {name:<10}: {'MATCH' if matches else 'MISMATCH'}")
    extra.append("")
    extra.append("Unit tests")
    extra.append(f" {'PASSED' if test_run.returncode == 0 else 'FAILED'} ({ran})")
    extra.append("")
    extra.append("Note for the risk team")
    extra.append("  Four outliers push the mean up to 22,400.00, while the median stays at 18,805.47.")
    extra.append("  Three of the four outliers come from agent AG-104.")
    extra.append("  Check AG-104's deposits first, and use the median to judge a normal day.")

    report_text = analyzer.report() + "\n" + "\n".join(extra)
    print(report_text)


    output_path = Path("output") / "stats_report.txt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report_text, encoding="utf-8")
    print(f"Report saved to {output_path}")