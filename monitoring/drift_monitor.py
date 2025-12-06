
from evidently.metrics import Report
from evidently.metrics import DataDriftTable

import pandas as pd

def generate_drift_report(reference, current, output_html="drift_report.html"):
    report = Report(metrics=[DataDriftTable()])
    report.run(reference_data=reference, current_data=current)
    report.save_html(output_html)
    return output_html
