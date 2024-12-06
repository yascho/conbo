import numpy as np
from statsmodels.stats.proportion import proportion_confint


def binomial_bounds(samples: np.ndarray, alpha=0.01):
    """
    Provides conservative Clopper-Pearson confidence bounds
    for success probability p.

    Parameters:
    ----------
    samples : np.ndarray
        An array of sample data. Can be a 1D array of binary outcomes
        or a 2D array where each row represents a set of binary outcomes.
    alpha : float, optional
        Significance level for the confidence interval. Default is 0.01.

    Notes:
    ------
    If only one bound is used (either lower or upper), pass 2 * alpha
    as the significance level to maintain the correct confidence level.

    Returns:
    -------
    tuple
        Lower and upper confidence bounds on success probability p
        that hold with probability at least 1-alpha.

    References:
    ----------
    Clopper, C. J., & Pearson, E. S. (1934). The use of confidence or fiducial
    limits illustrated in the case of the binomial. Biometrika, 26(4), 404-413.
    """
    num_samples = samples.shape[1] if samples.ndim == 2 else len(samples)
    return proportion_confint(samples.sum(-1), num_samples,
                              alpha=alpha, method="beta")


def cdf_bounds(samples: np.ndarray, x: np.ndarray, alpha=0.01):
    """
    Compute non-parametric, distribution-free confidence bounds
    for the cumulative distribution function (CDF)
    using the Dvoretzky-Kiefer-Wolfowitz (DKW) inequality.

    Parameters:
    -----------
    samples : np.ndarray
        Array of sample data. Can be 1D or 2D.
        If 2D, the function assumes samples are in columns
        (rows can be samples from potentially different distributions).
    x : np.ndarray
        Points at which to evaluate the CDF and its bounds.
    alpha : float, optional
        Significance level for the confidence bounds. Default is 0.01.

    Returns:
    --------
    cdf : np.ndarray
        Empirical CDF evaluated at points x.
    cdf_lower : np.ndarray
        Lower confidence bounds for the CDF.
    cdf_upper : np.ndarray
        Upper confidence bounds for the CDF.

    Notes:
    ------
    The DKW inequality guarantees that the empirical CDF is a close
    approximation to the actual CDF within an epsilon margin with
    probability 1 - alpha. The same epsilon holds for all points x.

    If only one bound is used (either lower or upper), pass 2 * alpha
    as significance level to maintain the correct confidence level.

    References:
    -----------
    Dvoretzky, A., Kiefer, J., & Wolfowitz, J. (1956). Asymptotic minimax
    character of the sample distribution function and of the classical
    multinomial estimator. The Annals of Mathematical Statistics, 27(3),
    642-669.
    """
    num_samples = samples.shape[1] if samples.ndim == 2 else len(samples)
    eps = np.sqrt(np.log(2/alpha)/(2*num_samples))

    cdf = np.column_stack([(samples <= i).mean(-1) for i in x]).squeeze()
    cdf_lower = np.maximum(cdf-eps, 0)
    cdf_upper = np.minimum(cdf+eps, 1)
    return cdf, cdf_lower, cdf_upper


def survival_bounds(samples: np.ndarray, x: np.ndarray, alpha=0.01):
    """
    Compute non-parametric, distribution-free confidence bounds for the
    survival function using the Dvoretzky-Kiefer-Wolfowitz (DKW) inequality.

    Parameters:
    -----------
    samples : np.ndarray
        Array of sample data. Can be 1D or 2D.
        If 2D, the function assumes samples are in columns
        (rows can be samples from potentially different distributions).
    x : np.ndarray
        Points at which to evaluate the survival function and its bounds.
    alpha : float, optional
        Significance level for the confidence bounds. Default is 0.01.

    Returns:
    --------
    survival : np.ndarray
        Empirical survival function evaluated at points x.
    survival_lower : np.ndarray
        Lower confidence bounds for the survival function.
    survival_upper : np.ndarray
        Upper confidence bounds for the survival function.

    Notes:
    ------
    Survival function is 1-CDF.

    If only one bound is used (either lower or upper), pass 2 * alpha
    as significance level to maintain the correct confidence level.
    """
    cdf, cdf_lower, cdf_upper = cdf_bounds(samples, x, alpha)
    return 1-cdf, 1-cdf_upper, 1-cdf_lower


def expectation_bounds(samples: np.ndarray, alpha=0.01, stepsize=100):
    """
    Distribution-free confidence bounds on the expectation of
    a random variable using Monte Carlo samples from the given distribution.
    Bounds are computed using the CDF-based nonparametric confidence interval.

    Parameters:
    -----------
    samples : np.ndarray
        An array of sample data points. Can be 1D or 2D.
        If 2D, the function assumes samples are in columns
        (rows can be samples from potentially different distributions).
    alpha : float, optional
        The significance level for the confidence interval (default is 0.01).
    stepsize : int, optional
        The number of steps to use in the Riemann sum approximation
        (default is 100).

    Notes:
    ------
    This function computes expectation bounds for a given set of samples
    assuming the support of the distribution is [0, 1].

    If only one bound is used (either lower or upper), pass 2 * alpha
    as significance level to maintain the correct confidence level.

    Returns:
    --------
    tuple
        A tuple containing the mean of the samples,
        lower and upper bounds on the expectation of the random variable.

    References:
    -----------
    Theodore Wilbur Anderson. Confidence limits for the expected value of an
    arbitrary bounded random variable with a continuous distribution function.
    Bulletin of The International and Statistical Institute, 43:249-251, 1969.
    """
    assert samples.min() >= 0 and samples.max() <= 1, \
        "Support of the distribution must be in [0, 1]."

    # https://en.wikipedia.org/wiki/CDF-based_nonparametric_confidence_interval
    T = np.linspace(0, 1, stepsize)
    cdf, cdf_lower, cdf_upper = cdf_bounds(samples, T, alpha)

    # https://en.wikipedia.org/wiki/Riemann_sum
    h = np.diff(T)
    exp_lower = 1-(cdf_upper[..., 1:]*h).sum(-1)  # right rule
    exp_upper = 1-(cdf_lower[..., :-1]*h).sum(-1)  # left rule
    return samples.mean(-1), exp_lower, exp_upper


def variance_bounds(samples: np.ndarray, alpha=0.01, stepsize=100):
    """
    Distribution-free confidence bounds on variance of a random variable
    using Monte Carlo samples from the given distribution. Bounds are computed
    using nonparametric confidence bounds on CDF.

    Parameters:
    -----------
    samples : np.ndarray
        An array of sample data points. Can be 1D or 2D.
        If 2D, the function assumes samples are in columns
        (rows can be samples from potentially different distributions).
    alpha : float, optional
        The significance level for the confidence interval (default is 0.01).
    stepsize : int, optional
        The number of steps to use in the approximation (default is 100).

    Notes:
    ------
    This function computes variance bounds for a given set of samples
    assuming the support of the distribution is bounded to [0, 1].

    Warning:
    --------
    Even if using only one bound (lower or upper), do NOT pass 2 * alpha
    as significance level since this will not maintain the correct
    confidence level (variance bounds use upper and lower CDF bounds).

    Returns:
    --------
    tuple
        A tuple containing the mean of the samples,
        lower and upper bounds on the expectation of the random variable.

    References:
    -----------
    Jan Schuchardt, Tom Wollschläger, Aleksandar Bojchevski, and
    Stephan Günnemann. Localized randomized smoothing for collective
    robustness certification. In ICLR 2023.
    """
    assert samples.min() >= 0 and samples.max() <= 1, \
        "Support of the distribution must be in [0, 1]."

    exp, exp_lower, exp_upper = expectation_bounds(samples, alpha=alpha)

    T = np.linspace(0, 1, stepsize)
    taus1 = T[:-1]
    taus2 = T[1:]
    if samples.ndim == 2:
        taus1 = np.expand_dims(taus1, 1).repeat(samples.shape[0], axis=1)
        taus2 = np.expand_dims(taus2, 1).repeat(samples.shape[0], axis=1)
    etas = np.stack([(taus1 - exp_lower)**2, (taus1-exp_upper)**2,
                    (taus2 - exp_lower)**2, (taus2-exp_upper)**2])

    etas_max = etas.max(axis=0).T
    etas_min = etas.min(axis=0).T

    cdf, cdf_lower, cdf_upper = cdf_bounds(samples, T, alpha=alpha)
    num_samples = samples.shape[1] if samples.ndim == 2 else len(samples)
    eps = np.sqrt(np.log(2/alpha)/(2*num_samples))

    # second inequality
    diff = (etas_max[..., :-1] - etas_max[..., 1:])
    var_upper = (cdf[..., 1:-1] + (2*(diff > 0)-1)*eps)*diff
    var_upper = var_upper.sum(-1)
    var_upper += etas_max[..., -1] - etas_max[..., 0]*cdf_lower[..., 0]

    diff = (etas_min[..., :-1] - etas_min[..., 1:])
    var_lower = (cdf[..., 1:-1] - (2*(diff > 0)-1)*eps)*diff
    var_lower = var_lower.sum(-1)
    var_lower += etas_min[..., -1] - etas_min[..., 0]*cdf_upper[..., 0]
    var_lower = np.maximum(var_lower, 0)

    sample_variance = (samples**2).mean(-1) - samples.mean(-1)**2
    sample_variance = np.maximum(sample_variance, 0)
    return sample_variance, var_lower, var_upper


def std_bounds(samples: np.ndarray, alpha=0.01, stepsize=100):
    """
    Distribution-free confidence bounds on the standard deviation of
    a random variable using Monte Carlo samples from the given distribution.

    Parameters:
    -----------
    samples : np.ndarray
        An array of sample data points. Can be 1D or 2D.
        If 2D, the function assumes samples are in columns
        (rows can be samples from potentially different distributions).
    alpha : float, optional
        The significance level for the confidence interval (default is 0.01).
    stepsize : int, optional
        The number of steps to use in the approximation (default is 100).

    Notes:
    ------
    This function computes variance bounds for a given set of samples
    assuming the support of the distribution is in [0, 1].

    Warning:
    --------
    Even if using only one bound (lower or upper), do NOT pass 2 * alpha
    as significance level since this will not maintain the correct
    confidence level (variance bounds use upper and lower CDF bounds).

    Returns:
    --------
    tuple
        A tuple containing the mean of the samples,
        lower and upper bounds on the expectation of the random variable.
    """
    params = samples, alpha, stepsize
    sample_variance, var_lower, var_upper = variance_bounds(*params)
    return np.sqrt(sample_variance), np.sqrt(var_lower), np.sqrt(var_upper)
