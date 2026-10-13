"""Bite2Text — orthodontic report generation from intraoral scans and photographs.

    from bite2text import ReportGenerator

    generator = ReportGenerator()                 # models load lazily, once
    report = generator(upper="ios_upper.stl",
                       lower="ios_lower.stl",
                       photos=["intraoral_1.jpg", ..., "intraoral_5.jpg"])
    print(report.text)
"""

from .pipeline import Report, ReportGenerator, generate_report

__version__ = "1.0.0"

__all__ = ["Report", "ReportGenerator", "generate_report", "__version__"]
