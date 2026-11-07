Day 3: Statistical Amalysis Toolkit 
        OVERVIEW
This project looks at the daily deposits in KES of mobile-money agents and works out simple statistics about them. it finds unusual values (outliess), rejected bad data and answers against NumPy.

        FILES 
stats_toolkit.py
tests_stats_toolkit.py
data/agent_deposits.csv
output/stats_report.txt
requirements,txt

        REQUIREMENTS
python 3.1
NumPy (only check results)
pytest (optionsl)
--install using pip install -r requirements.txt

        APPROACH
Load the deposits column from the csv file.
Reject misssing and invalid values, keeping reasons for each.
Calculate the mean, median, mode, variance and std deviation.
Calculate the quartiles and the IQR.
Find outliers with the IQR rile.
Show all data and outliers-trmoved side by side.
Check results against NumPy.

        MY CHOICES
I used sample formula, which divides by n - 1. the 40 days are only a small part of what the agents do over time, so the data is a sample.
I also used linear interpolation. For n sorted values, the quartile for a fraction p is at position (n - 1) * p, counting from 0. If that position falls between two values, I takke a point betweeen them.

        RESULTS
    40 rows were read 
    35 rows were valid
    5 rows were rejected (rows: 6, 9, 13, 24 and 28)

The outlier fences are 8,530.19 and 28,445.41. There are four outliers:
    96,500.00 (2026-09-20, AG-104)
    1,200.00 (2026-10-02, AG-104)
    88,400.00 (2026-10-06, AG-104)
    29060.99 (2026-10-07, AG-104)

The outliers puth the mean up and make the standard deviation much larger. 

        CHECKING WITH NUMPY
compare_with_numpy() recalculates the mean, median, variance, standard deviation and quartiles with NumPy. All the five resilts match mine.