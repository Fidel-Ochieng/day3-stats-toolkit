&#x09;Day 3: Statistics Toolkit 

Overview 

A python project that analyses daily agents deposits in (KES). 

&#x20;It :   calculates descriptive statistics

&#x09; outliers,

&#x09; clean invalid data 

&#x09; compares its results with NumPy 



Approach 

1. Load data from a csv file.
2. Handled missing and invalid values.
3. Calculated mean, median mode, variance, and standard deviation.
4. Calculated quartiles and interquartile range (IQR)
5. Detected outliers using the IQR method.
6. Compared statistical results with NumPy.
7. Saved the statistics report to a text file.



Requirements

Python 3.10 or latest 

NumPy 



Installation 

install NumPy using: 

&#x20;pip install -r requirement.txt 



Key Findings 

1.Data processed: 40 rows were read

&#x09;	  35 valid values

&#x09;	  5 rejected entries

2.Average deposit: KES22,400.00

3.Median deposit: KES18,805.47

4.Outliers detected: 4 unusual deposit values were identified using the IQR method

5.Cleaned data: Removing the outliers left 31 values, with an average deposit of KES 18,349.65

6.Data validation: Missing values and invalid entries were rejected.

7.Verification: The calculated statistical results matched NumPy's results









