# SampleParaCal


[![PyPI version](https://badge.fury.io/py/medsamplesize.svg)](https://badge.fury.io/py/spcal)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive Python package for calculating sample sizes in medical diagnostic studies. This package implements a wide range of statistical methods for diagnostic research, including:

- Sample size calculations for diagnostic accuracy studies
- ROC curve analysis
- Comparison of multiple diagnostic tests
- Non-inferiority and equivalence testing
- Multi-reader multi-case (MRMC) study design

## Installation

```bash
pip install spcal
```

## Quick Start

```python
import spcal as spc

# Calculate sample size for a study using a two-sided confidence interval
n = spc.two_sided_CI_sample_size(var=0.25, alpha=0.05, L=0.1)
print(f"Required sample size: {n}")

# Calculate AUC variance
variance = spc.AUC_variance_binormal(A=0.85, R=1.5)
print(f"AUC variance: {variance:.6f}")

# Compare two diagnostic tests
n = spc.sample_size_for_two_diagnostic_tests(
    alpha=0.05, 
    beta=0.2, 
    delta=0.1, 
    Se1=0.85, 
    Se2=0.75, 
    coPos=0.6
)
print(f"Sample size for comparison: {n}")
```

## Features

### Single Diagnostic Method Evaluation

- Confidence interval-based sample size calculations
- Area Under ROC Curve (AUC) variance estimation
- Partial AUC analysis
- High-accuracy test evaluation
- Clustered data analysis

### Threshold Optimization

- Sensitivity calculation at fixed false positive rates
- Variance estimation for transformed sensitivity
- Binormal ROC curve modeling

### Diagnostic Method Comparison

- Paired and unpaired sample size calculations
- Relative sensitivity and specificity comparisons
- Covariance estimation for correlated tests
- Predictive value comparison (PPV/NPV)

### Non-inferiority and Equivalence Testing

- Non-inferiority sample size calculations
- Equivalence testing for diagnostic methods
- Clustered data equivalence testing

### Multi-reader Studies

- Variance components for reader variability
- Sample size for multi-reader studies
- Multi-reader multi-case (MRMC) study design

[//]: # (## Documentation)

[//]: # ()
[//]: # (Detailed API documentation with examples is available at [documentation link].)

[//]: # ()
[//]: # (Each function includes detailed parameter descriptions, mathematical formulas, and usage examples to guide researchers in selecting appropriate methods for their study designs.)

[//]: # (## Citation)

[//]: # ()
[//]: # (If you use this package in your research, please cite:)

[//]: # ()
[//]: # (```)

[//]: # ([Citation information])

[//]: # (```)

## License

This project is licensed under the MIT License - see the LICENSE file for details.

[//]: # ()
[//]: # (## Contributing)

[//]: # ()
[//]: # (Contributions are welcome! Please feel free to submit a Pull Request.)
[Demo](https://spcal-demo.vercel.app/)
[Book Reference](https://onlinelibrary.wiley.com/doi/book/10.1002/9780470906514)