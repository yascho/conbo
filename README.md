<img src="./confidence-interval-visualization.svg">

# ConBo: Bounding properties of random variables with high confidence

ConBo (**Confidence Bounds**) is the ultimate toolkit for computing distribution-free, non-parametric bounds on properties of random variables with high confidence. It is particularly useful in scenarios where no assumptions can be made about the underlying distribution, for example when working with complex models such as large language models (LLMs) or flow-based generative models.

## Installation

To install the required dependencies, run:

```bash
pip install conbo
```

## Usage

To use the confidence bounds, import functionality from the `conbo` module and call the desired functions. Here is an example:

```python
from conbo import *

# Example usage
cdf, cdf_lower, cdf_upper = cdf_bounds(samples, x=[0.5])
sample_mean, exp_lower, exp_upper = expectation_bounds(samples)
sample_variance, var_lower, var_upper = variance_bounds(samples)
sample_std, std_lower, std_upper = std_bounds(samples)
```

## Examples

You can find usage examples in the [demo notebook](demo.ipynb) demonstrating how to compute the confidence bounds.


## Cite

This repository implements confidence bounds discussed in the paper [**A Probabilistic Perspective on Unlearning and Alignment for Large Language Models**](https://arxiv.org/abs/2410.03523) by Yan Scholten, Stephan Günnemann, and Leo Schwinn. Please cite our paper if you use this code in your own work:

```
@misc{scholten2024probabilistic,
      title={A Probabilistic Perspective on Unlearning and Alignment for Large Language Models},
      author={Yan Scholten and Stephan Günnemann and Leo Schwinn},
      year={2024},
      eprint={2410.03523},
      archivePrefix={arXiv},
      url={https://arxiv.org/abs/2410.03523},
}
```

## Contact

For questions and feedback please contact:

Yan Scholten, Technical University of Munich<br>
Stephan Günnemann, Technical University of Munich<br>
Leo Schwinn, Technical University of Munich

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.