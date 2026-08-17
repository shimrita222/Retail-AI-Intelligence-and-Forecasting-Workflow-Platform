# Data Analyst Crew — Insights

## Data Profile

- **29,518 rows**, 3 stores, 78 departments, date range 2010-02-05 to 2012-10-26.
- Null counts: MarkDown1=18953, MarkDown2=21001, MarkDown3=19532, MarkDown4=19084, MarkDown5=18953

## Micro-Inspection Findings (record-level, not aggregate)

- **Zero-sales records:** 6 (sample: [{'Store': 1, 'Dept': 47, 'Date': '2011-03-11', 'Weekly_Sales': 0.0}, {'Store': 1, 'Dept': 47, 'Date': '2011-08-12', 'Weekly_Sales': 0.0}, {'Store': 1, 'Dept': 47, 'Date': '2011-08-19', 'Weekly_Sales': 0.0}])
- **Negative-sales records:** 46 (sample: [{'Store': 2, 'Dept': 47, 'Date': '2010-10-15', 'Weekly_Sales': -1098.0}, {'Store': 2, 'Dept': 47, 'Date': '2010-11-19', 'Weekly_Sales': -1098.0}, {'Store': 2, 'Dept': 47, 'Date': '2010-07-30', 'Weekly_Sales': -1098.0}])
- **Top holiday spikes per individual store:**
  - Store 3: holiday mean 6916.45 vs non-holiday mean 6332.10 (ratio 1.09x)
  - Store 2: holiday mean 28798.71 vs non-holiday mean 26753.86 (ratio 1.08x)
  - Store 1: holiday mean 23039.39 vs non-holiday mean 21609.63 (ratio 1.07x)
- **Extreme markdown outliers (record-level):**
  - MarkDown1 = 75149.79 at Store 2, Dept 79, 2012-02-03 (upper bound 27952.39)
  - MarkDown1 = 75149.79 at Store 2, Dept 80, 2012-02-03 (upper bound 27952.39)
  - MarkDown1 = 75149.79 at Store 2, Dept 81, 2012-02-03 (upper bound 27952.39)
  - MarkDown1 = 75149.79 at Store 2, Dept 82, 2012-02-03 (upper bound 27952.39)
  - MarkDown1 = 75149.79 at Store 2, Dept 83, 2012-02-03 (upper bound 27952.39)
- **Largest week-over-week anomalies:**
  - Store 1, Dept 47, 2011-10-14: 0.00 -> -498.00 (-inf%)
  - Store 1, Dept 47, 2011-04-08: 0.00 -> -298.00 (-inf%)
  - Store 2, Dept 60, 2010-05-07: 0.00 -> 6.00 (inf%)
  - Store 3, Dept 36, 2012-08-24: 0.00 -> 24.00 (inf%)
  - Store 2, Dept 47, 2012-07-06: 0.00 -> -28.00 (-inf%)

## Business Intelligence

- Holiday-week mean sales: 20129.09 vs non-holiday mean: 18715.15.
- Mean sales in weeks with any markdown active: 19301.03 vs without: 18543.59 (15 extreme markdown outliers detected).
- Monthly trend spans 2010-02 (19430.66) to 2012-10 (18643.52).

## Data Contract

- Primary keys: ['Store', 'Dept', 'Date']; target: `Weekly_Sales` bounded to [-5000.0, 800000.0].
