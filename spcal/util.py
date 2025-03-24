#
# import math
# from scipy.optimize import fsolve
# from scipy.stats import norm
#
#
# def one_sided_ci_sample_size(alpha, beta, var, L):#6.1
#     z_a = norm.ppf(alpha / 2)
#     z_b = norm.ppf(beta)
#     n = (z_a + z_b) ** 2 * var / (L ** 2)
#     return math.ceil(n)
#
# def two_sided_ci_sample_size(var,alpha,L):#6.2
#     z_a=norm.ppf(alpha/2)
#     n=z_a**2*var/(L**2)
#     return math.ceil(n)
# def total_sample_for_disease_prevalence(n,prev_p,beta):#6.3
#     def equation(N_total):
#         z_b=-norm.ppf(beta)
#         N_total=N_total[0]
#         return (N_total*prev_p-n)/math.sqrt(N_total * prev_p * (1 - prev_p)) - z_b
#     N_total_init=n/prev_p
#     N_total_res=fsolve(equation,[N_total_init])
#     return math.ceil(N_total_res[0])
# def roc_variance_exponential(A,R):#6.4
#     Q1=A/(2-A)
#     Q2=2*A**2/(1+A)
#     var=Q1/R+Q2-A**2*(1/R+1)
#     return var
# def roc_variance_binormal(A,R):#6.5
#     a=1.414*norm.ppf(A)
#     var=0.0099*math.exp(-a**2/2)*(5*a**2+8+(a**2+8)/R)
#     return var
# def roc_variance_universal(A):#6.6
#     var=A*(1-A)
#     return var
# def clustered_sample_size(n,s,r):#6.7
#     nc=n*(1+r*(s-1))
#     return nc
# def roc_accuracy_hypothesis_sample_size(alpha,beta,V0,VA,t0,t1):#6.8
#     z_a=norm.ppf(alpha/2)
#     z_b=norm.ppf(beta)
#     n=(z_a*math.sqrt(V0)+z_b*math.sqrt(VA))**2/(t1-t0)**2
#     return n
# def transformed_sensitivity_at_fixed_fpr(a, b, e):#6.9
#     """计算灵敏度的标准化 z 变换 (公式 6.9)"""
#     return a + b * norm.ppf(e)
# def variance_transformed_sensitivity(a, b, e, R):#6.10
#     """计算 V(z(Sep_{FPR=f})) 的方差 (公式 6.10)"""
#     g=norm.ppf(e)
#     return 1 + b**2/R +a ** 2 /2+ g**2 * b ** 2 * (1 + R) / (2 * R)
# def transformed_sensitivity_at_fixed_fpr(b, FPR, Se):#6.11
#     """
#     计算公式 (6.11) 左侧的 a 值
#
#     :param b: 经验参数 b
#     :param FPR: 假阳性率 (False Positive Rate)
#     :param Se: 敏感度 (Sensitivity)
#     :return: 计算得到的 a 值
#     """
#     # 计算 Φ⁻¹(1.0 - FPR)
#     z_fpr = norm.ppf(1.0 - FPR)
#
#     # 计算 Φ⁻¹(1.0 - Se)
#     z_sensitivity = norm.ppf(1.0 - Se)
#
#     # 计算 a
#     a = b * z_fpr - z_sensitivity
#
#     return a
#
# def variance_of_partial_roc_area(a, b, e1, e2, R):#6.12
#     """
#     计算部分 ROC 曲线面积 (pAUC) 在 FPR 范围 e1 到 e2 内的方差 (公式 6.12)
#
#     :param a: binormal 参数 a
#     :param b: binormal 参数 b
#     :param e1: FPR 下界
#     :param e2: FPR 上界
#     :param R: 样本比例 (患者 vs 非患者)
#     :return: 部分 AUC 的方差
#     """
#
#     # 计算 e' 和 e''
#     e1_prime = (norm.ppf(e1) + (a * b) * (1 + b ** 2) ** (-1)) * math.sqrt(1 + b ** 2)
#     e2_prime = (norm.ppf(e2) + (a * b) * (1 + b ** 2) ** (-1)) * math.sqrt(1 + b ** 2)
#
#     e1_double_prime = (e1_prime ** 2) / 2
#     e2_double_prime = (e2_prime ** 2) / 2
#
#     # 计算各个 expr
#     expr1 = math.exp(-a ** 2 / (2 * (1 + b ** 2)))
#     expr2 = (1 + b ** 2)
#     expr3 = norm.cdf(e2_prime) - norm.cdf(e1_prime)
#     expr4 = math.exp(-e1_double_prime) - math.exp(-e2_double_prime)
#
#     # 计算 f
#     f = expr1 * (1 / math.sqrt(2 * math.pi * expr2)) * expr3
#     print(f)
#     # 计算 g
#     g = expr1 * (1 / (2 * math.pi * expr2)) * expr4 - (a * b) * expr1 * ( 2 * math.pi * expr2 ** 3)**(-0.5) * expr3
#     print(g)
#     # 计算 V(A_{e1 <= FPR <= e2})
#     variance = f ** 2 * (1 + b ** 2 / R + a ** 2 / 2) + g ** 2 * (b ** 2 * (1 + R) / (2 * R))
#
#     return variance
#
#
# def sample_size_for_two_diagnostic_tests(alpha, beta, delta, Se1, Se2, coPos):
#     """
#     计算诊断测试样本量 (公式 6.13 - 6.16)
#
#     :param alpha: 显著性水平 (Type I error rate)
#     :param beta: 统计效能 (Type II error rate, 1 - Power)
#     :param theta1: 诊断测试 1 的准确度 (θ1)
#     :param theta2: 诊断测试 2 的准确度 (θ2)
#     :param Se1: 备择假设下测试 1 的灵敏度
#     :param Se2: 备择假设下测试 2 的灵敏度
#     :param copos: P(T1 = 1 | T2 = 1)，测试 1 结果为阳性的概率 (给定测试 2 结果为阳性)
#     :return: 样本量 n
#     """
#     # 计算 ψ (公式 6.16)
#     psi = Se1 + Se2 - 2 * Se2 * coPos
#
#     # 计算 V_o 和 V_A (公式 6.15)
#
#     Vo = psi
#     VA = psi - delta ** 2
#
#     # 计算样本量 (公式 6.13)
#     z_alpha = norm.ppf(1 - alpha / 2)
#     z_beta = norm.ppf(1 - beta)
#     numerator = (z_alpha * math.sqrt(Vo) + z_beta * math.sqrt(VA)) ** 2
#     n = numerator / (delta ** 2)
#
#     return math.ceil(n)
#
#
# def unpaired_sample_size(n1, n):
#     """
#     计算非配对研究中的未知样本量 n2 (公式 6.17)
#
#     :param n1: 已知测试的固定样本量
#     :param n: 由公式 (6.13) 计算出的理论样本量
#     :return: 计算得到的未知测试样本量 n2
#     """
#     return math.ceil((n * n1) / (2 * n1 - n))
#
# def sample_size_rPPV(alpha, beta, gamma, delta, p5, p6, p7, p3, PPV2):
#     """
#     计算 rPPV 相关的样本量 (公式 6.18)
#     """
#     z_alpha = norm.ppf(1 - alpha / 2)
#     z_beta = norm.ppf(1 - beta)
#     log_term = math.log(gamma / delta,math.e)
#     term1 = (z_beta + z_alpha) ** 2 / (log_term ** 2)
#     term2 = 1 / ((p5 + p6) * (p5 + p7))
#     term3 = 2 * (p7 + p3) * gamma * PPV2 ** 2 + (-p6 + p5 * (1 - gamma)) * PPV2 + p6 + p7 * (1 - 3 * gamma * PPV2)
#     n = term1 * term2 * term3
#     return math.ceil(n)
#
# def sample_size_rNPV(alpha, beta, gamma, delta, p2, p4, p8, p3, NPV2):
#     """
#     计算 rNPV 相关的样本量 (公式 6.19)
#     """
#     z_alpha = norm.ppf(1 - alpha / 2)
#     z_beta = norm.ppf(1 - beta)
#
#     log_term = math.log(gamma / delta)
#     print(log_term)
#     term1 = (z_beta + z_alpha) ** 4 / (log_term ** 2)
#     term2 = 1 / ((p2 + p4) * (p3 + p4))
#     term3 = -2 * (p4 + p8) * gamma * NPV2 ** 2 + (-p3 + p4 - gamma * (p2 - p4)) * NPV2 + p2 + p3
#     n = term1 * term2 * term3
#     return math.ceil(n)
#
# def covariance_for_two_roc(a1, a2, rD, rN, R):
#     """
#     计算 Ĉ(Â1, Â2) (公式 6.20)
#     """
#     term1 = math.exp(-(a1**2 + a2**2) / 4) / 12.5664 * (rD + rN / R + (rD**2 * a1 * a2) / 2)
#     term2 = math.exp(-(a1**2 + a2**2) / 4) / 50.2655 * ((a1 * a2 * (rN**2 + R * rD**2)) / (2 * R))
#     term3 = math.exp(-(a1**2 + a2**2) / 4) / 25.1327 * (rD**2 * a1 * a2)
#     return term1 + term2 - term3
# def variance_compare_two_tests(theta1, theta2, r):
#     """
#     计算 V_o 和 V_A (公式 6.21)
#     """
#     Vo = theta1 * (1 - theta1) + theta1 * (1 - theta1) - 2 * r * math.sqrt(theta1 * (1 - theta1) * theta1 * (1 - theta1))
#     VA = theta1 * (1 - theta1) + theta2 * (1 - theta2) - 2 * r * math.sqrt(theta1 * (1 - theta1) * theta2 * (1 - theta2))
#     return Vo, VA
#
# def covariance_compare_sensitivity_fixed_fpr(a1, a2, b1, b2, rD, rN, R, e):
#     """
#     计算 Ĉ((Ŝ_{e_{FPR=e}})₁, (Ŝ_{e_{FPR=e}})₂) (公式 6.22)
#     """
#     g = norm.ppf(e)
#     term1 = rD + (rN * b1 * b2) / R + (rD**2 * a1 * a2) / 2
#     term2 = (g**2 * b1 * b2 * (rN**2 + R * rD**2)) / (2 * R)
#     term3 = (g * rD**2 / 2) * (a1 * b2 + a2 * b1)
#     return term1 + term2 + term3
#
#
# def compute_f_g(a, b, e1, e2):
#     """
#     计算 f 和 g (参考公式 6.12)
#     """
#
#     e1_prime = (norm.ppf(e1) + (a * b) * (1 + b ** 2) ** (-1)) * math.sqrt(1 + b ** 2)
#     e2_prime = (norm.ppf(e2) + (a * b) * (1 + b ** 2) ** (-1)) * math.sqrt(1 + b ** 2)
#
#     e1_double_prime = (e1_prime ** 2) / 2
#     e2_double_prime = (e2_prime ** 2) / 2
#
#     # 计算各个 expr
#     expr1 = math.exp(-a ** 2 / (2 * (1 + b ** 2)))
#     expr2 = (1 + b ** 2)
#     expr3 = norm.cdf(e2_prime) - norm.cdf(e1_prime)
#     expr4 = math.exp(-e1_double_prime) - math.exp(-e2_double_prime)
#
#     # 计算 f
#     f = expr1 * (1 / math.sqrt(2 * math.pi * expr2)) * expr3
#     print(f)
#     # 计算 g
#     g = expr1 * (1 / (2 * math.pi * expr2)) * expr4 - (a * b) * expr1 * (2 * math.pi * expr2 ** 3) ** (-0.5) * expr3
#     return f, g
#
#
# def covariance_partial_AUC(a1, a2, b1, b2, e1, e2, rD, rN, R):
#     """
#     计算 Ĉ((Â_{e₁≤FPR≤e₂})₁, (Â_{e₁≤FPR≤e₂})₂) (公式 6.23)
#     """
#     f1, g1 = compute_f_g(a1, b1, e1, e2)
#     f2, g2 = compute_f_g(a2, b2, e1, e2)
#
#     term1 = f1 * f2 * (rD + (rN * b1 * b2) / R + (rD ** 2 * a1 * a2) / 2)
#     term2 = g1 * g2 * (b1 * b2 * (rN ** 2 + R * rD ** 2)) / (2 * R)
#     term3 = f1 * g2 * (rD ** 2 * a1 * b2 / 2) + f1 * g2 * (rD ** 2 * a2 * b1)
#
#     return term1 + term2 + term3
#
# def non_inferiority_sample_size(alpha, beta, theta_S, theta_E, delta_M, var):
#     """
#     计算非劣效性检验的样本量 (公式 6.24)
#     """
#     z_alpha = norm.ppf(1 - alpha)
#     z_beta = norm.ppf(1 - beta)
#     numerator = (z_alpha + z_beta) ** 2 * var
#     denominator = (theta_S - theta_E - delta_M) ** 2
#     n = numerator / denominator
#     return math.ceil(n)
#
#
# def equivalence_sample_size(alpha, beta, theta_S, theta_E, delta_L, delta_U, var):
#     """
#     计算等效性检验的样本量 (公式 6.25 - 6.27)
#     """
#     z_alpha = norm.ppf(1 - alpha)
#     z_beta = norm.ppf(1 - beta)
#     diff = theta_S - theta_E
#
#     if diff > 0:
#         denominator = (delta_U - diff) ** 2
#     elif diff < 0:
#         denominator = (delta_U + diff) ** 2
#     else:
#         z_beta /= 2
#         denominator = delta_U ** 2
#
#     numerator = (z_alpha + z_beta) ** 2 * var
#     n = numerator / denominator
#     return math.ceil(n)
#
# def relative_tpr_fpr(TPR1, TPR2, FPR1, FPR2):
#     """
#     计算相对真阳性率 (rTPR) 和相对假阳性率 (rFPR) (公式 6.28)
#     """
#     rTPR = TPR1 / TPR2
#     rFPR = FPR1 / FPR2
#     return rTPR, rFPR
# def rTPR_sample_size(alpha, beta, gamma, delta_1, TPR1, TPR2, TPPR):
#     """
#     计算 rTPR 相关的样本量 (公式 6.29)
#     """
#     z_beta = norm.ppf(1 - beta)
#     alpha_star = 1 - math.sqrt(1 - alpha)
#     z_alpha_star = norm.ppf(1 - alpha_star)
#     log_term = math.log(gamma / delta_1)
#     term1 = (z_beta + z_alpha_star) ** 2 / (log_term ** 2)
#     term2 = ((gamma + 1) * TPR2 - 2 * TPPR) / (gamma * TPR2 ** 2)
#     n = term1 * term2
#     return math.ceil(n)
#
#
# def schafer_sample_size(alpha, beta, SP_prime, SE_SP, SE_prime, b):
#     """
#     计算 Schafer (1989) 方法的样本量上界 (公式 6.30)
#     """
#     lambda_val = norm.ppf(SE_SP) - norm.ppf(SE_prime)
#     vx = b * math.sqrt(1 + 0.5 * norm.ppf(SP_prime) ** 2)
#     vy = math.sqrt(1 + 0.5 * norm.ppf(SE_prime) ** 2)
#
#     numerator = (math.sqrt(2) * norm.ppf(math.sqrt(1 - alpha)) + norm.ppf(1 - beta)) ** 2 * (vx + vy) ** 2
#     denominator = lambda_val ** 2
#
#     N = numerator / denominator
#     return math.ceil(N)
# def noncentrality_parameter_for_multireader_study(J, v1, v2, sigma_b, rho_b, sigma_w, Q, sigma_c, rho_1, rho_2, rho_3):
#     """
#     计算 λ (公式 6.32)
#     """
#     numerator = J * (v1 - v2) ** 2
#     denominator = 2 * (sigma_b**2 * (1 - rho_b) + (sigma_w**2 / Q) + sigma_c**2 * ((1 - rho_1) + (J - 1) * (rho_2 - rho_3)))
#     return numerator / denominator
# def variance_for_multireader(Vo_theta, J, rho_dr):
#     """
#     计算固定读者 MRMC 设计的方差函数 (公式 6.33)
#     """
#     return Vo_theta * (1 / J + (J - 1) * rho_dr / J)
#
# def sample_size_for_multireader(alpha, beta, Vo_theta, VA_theta, v1, v2):
#     """
#     计算固定读者 MRMC 设计的样本量 (公式 6.34)
#     """
#     z_alpha = norm.ppf(1 - alpha / 2)
#     z_beta = norm.ppf(1 - beta)
#     numerator = (z_alpha * math.sqrt(Vo_theta) + z_beta * math.sqrt(VA_theta)) ** 2
#     denominator = (v1 - v2) ** 2
#     return math.ceil(numerator / denominator)
# def noncentrality_parameter_for_multireader_multicase_study(v1, v2, J, N, sigma_TR2, sigma_TP2, sigma2):
#     """
#     计算非中心性参数 λ (公式 6.35)
#     """
#     numerator = (v1 - v2) ** 2
#     denominator = (2 / (J * N)) * (N * sigma_TR2 + J * sigma_TP2 + sigma2)
#     return numerator / denominator
#
"""
Core functions for sample size calculation in diagnostic studies.

This module contains all the core functions for sample size calculation and related
statistical parameters in diagnostic medicine studies. Functions are organized by
categories: confidence interval methods, AUC analysis, ROC curve analysis, comparison
methods, and multi-reader studies.
"""

import math
from scipy.optimize import fsolve
from scipy.stats import norm


#
# Confidence Interval Methods
#

def one_sided_CI_sample_size(alpha, beta, var, L):
    """
    Calculate sample size for one-sided confidence interval of a diagnostic test parameter.

    This function is suitable when only a lower or upper limit on sensitivity or
    specificity is required. Used for scenarios where research focuses on a clear
    minimum requirement for a diagnostic indicator.

    Parameters
    ----------
    alpha : float
        Significance level (e.g., 0.05)
    beta : float
        Statistical power (1 - Type II error rate, e.g., 0.8)
    var : float
        Variance of the parameter of interest
    L : float
        Precision margin (half-width of the confidence interval)

    Returns
    -------
    int
        Required sample size, rounded up to the nearest integer

    Notes
    -----
    Formula: n = (Z_{α/2} + Z_β)² × Var / L²

    Examples
    --------
    >>> one_sided_CI_sample_size(alpha=0.05, beta=0.2, var=0.25, L=0.1)
    97
    """
    z_a = norm.ppf(alpha / 2)
    z_b = norm.ppf(beta)
    n = (z_a + z_b) ** 2 * var / (L ** 2)
    return math.ceil(n)


def sample_size_for_clustered_equivalence_testing(alpha, beta, theta_S, theta_E, delta_L, delta_U, var):
    """
    Calculate sample size for equivalence testing with clustered data.

    This function is used for equivalence studies with clustered observations when
    demonstrating equivalence between methods for correlated measurements.

    Parameters
    ----------
    alpha : float
        Significance level (e.g., 0.05)
    beta : float
        Statistical power (1 - Type II error rate, e.g., 0.8)
    theta_S : float
        Standard treatment effect
    theta_E : float
        Experimental treatment effect
    delta_L : float
        Lower equivalence margin
    delta_U : float
        Upper equivalence margin
    var : float
        Variance of the effect size

    Returns
    -------
    int
        Required sample size, rounded up to the nearest integer

    Notes
    -----
    Formula is similar to standard equivalence testing, but accounts for clustering:
        n = (z_α + z_β)²·var/{(δ_U - (θ_S - θ_E))² if θ_S - θ_E > 0, or
                               (δ_U + (θ_S - θ_E))² if θ_S - θ_E < 0, or
                               δ_U²/4 if θ_S - θ_E = 0}

    Examples
    --------
    >>> sample_size_for_clustered_equivalence_testing(alpha=0.05, beta=0.2, theta_S=0.85, theta_E=0.82, delta_L=-0.1, delta_U=0.1, var=0.06)
    92
    """
    z_alpha = norm.ppf(1 - alpha)
    z_beta = norm.ppf(1 - beta)
    diff = theta_S - theta_E

    if diff > 0:
        denominator = (delta_U - diff) ** 2
    elif diff < 0:
        denominator = (delta_U + diff) ** 2
    else:
        # Special case: difference is 0
        z_beta /= 2  # Adjust z_beta
        denominator = delta_U ** 2 / 4

    numerator = (z_alpha + z_beta) ** 2 * var
    n = numerator / denominator

    return math.ceil(n)


def rTPR_sample_size(alpha, beta, gamma, delta_1, TPR1, TPR2, TPPR):
    """
    Calculate sample size for comparing relative True Positive Rates.

    This function is used for non-inferiority or superiority testing based on
    detection rate ratios, when the primary outcome is the ratio of sensitivities.

    Parameters
    ----------
    alpha : float
        Significance level (e.g., 0.05)
    beta : float
        Statistical power (1 - Type II error rate, e.g., 0.8)
    gamma : float
        Ratio of two diagnostic test performance
    delta_1 : float
        Non-inferiority margin
    TPR1 : float
        True Positive Rate (TPR) of the first test
    TPR2 : float
        True Positive Rate (TPR) of the second test
    TPPR : float
        Proportion of positive predictive values

    Returns
    -------
    int
        Required sample size, rounded up to the nearest integer

    Notes
    -----
    Formulas:
        α* = 1 - √(1 - α)
        n = (Z_β + Z_{α*})²/(log(γ/δ₁))²·((γ+1)TPR₂ - 2TPPR)/(γ·TPR₂²)

    Examples
    --------
    >>> rTPR_sample_size(alpha=0.05, beta=0.2, gamma=1.2, delta_1=1.1, TPR1=0.85, TPR2=0.75, TPPR=0.7)
    122
    """
    z_beta = norm.ppf(1 - beta)
    alpha_star = 1 - math.sqrt(1 - alpha)
    z_alpha_star = norm.ppf(1 - alpha_star)

    log_term = math.log(gamma / delta_1)
    term1 = (z_beta + z_alpha_star) ** 2 / (log_term ** 2)
    term2 = ((gamma + 1) * TPR2 - 2 * TPPR) / (gamma * TPR2 ** 2)

    n = term1 * term2
    return math.ceil(n)


def schafer_sample_size(alpha, beta, SP_prime, SE_SP, SE_prime, b):
    """
    Calculate sample size using Schafer (1989) method for threshold optimization.

    This function applies Schafer's method to calculate sample size bounds for
    threshold optimization studies.

    Parameters
    ----------
    alpha : float
        Significance level (e.g., 0.05)
    beta : float
        Statistical power (1 - Type II error rate, e.g., 0.8)
    SP_prime : float
        Estimated specificity of the diagnostic test
    SE_SP : float
        Sensitivity-specificity correlation coefficient
    SE_prime : float
        Estimated sensitivity of the diagnostic test
    b : float
        Scaling factor for variance adjustment

    Returns
    -------
    int
        Required sample size, rounded up to the nearest integer

    Notes
    -----
    Formulas:
        λ = Φ⁻¹(SE_{SP}) - Φ⁻¹(SE')
        v_x = b·√(1 + 0.5·(Φ⁻¹(SP'))²)
        v_y = √(1 + 0.5·(Φ⁻¹(SE'))²)
        N = (√2·Φ⁻¹(√(1-α)) + Φ⁻¹(1-β))²·(v_x + v_y)²/λ²

    Reference: Schafer H. Constructing a cut-off point for a quantitative diagnostic test.
    Stat Med. 1989;8:1381-91.

    Examples
    --------
    >>> schafer_sample_size(alpha=0.05, beta=0.2, SP_prime=0.9, SE_SP=0.85, SE_prime=0.8, b=1.1)
    87
    """
    lambda_val = norm.ppf(SE_SP) - norm.ppf(SE_prime)
    vx = b * math.sqrt(1 + 0.5 * norm.ppf(SP_prime) ** 2)
    vy = math.sqrt(1 + 0.5 * norm.ppf(SE_prime) ** 2)

    numerator = (math.sqrt(2) * norm.ppf(math.sqrt(1 - alpha)) + norm.ppf(1 - beta)) ** 2 * (vx + vy) ** 2
    denominator = lambda_val ** 2

    N = numerator / denominator
    return math.ceil(N)


#
# Multi-reader Studies
#

def variance_for_multireader(Vo_theta, J, rho_dr):
    """
    Calculate variance for multi-reader studies.

    This function accounts for reader variability in diagnostic accuracy studies
    involving multiple readers interpreting diagnostic tests.

    Parameters
    ----------
    Vo_theta : float
        Variance of the test statistic
    J : int
        Number of readers
    rho_dr : float
        Correlation between diagnostic readings

    Returns
    -------
    float
        Variance for multi-reader study

    Notes
    -----
    Formula: V = Vo_θ·(1/J + (J-1)ρ_{dr}/J)

    Examples
    --------
    >>> variance_for_multireader(Vo_theta=0.04, J=5, rho_dr=0.3)
    0.0368
    """
    return Vo_theta * (1 / J + (J - 1) * rho_dr / J)


def sample_size_for_multireader(alpha, beta, Vo_theta, VA_theta, v1, v2):
    """
    Calculate sample size for multi-reader studies comparing diagnostic methods.

    This function is used for study designs with fixed readers evaluating
    multiple cases when comparing diagnostic methods with reader variability.

    Parameters
    ----------
    alpha : float
        Significance level (e.g., 0.05)
    beta : float
        Statistical power (1 - Type II error rate, e.g., 0.8)
    Vo_theta : float
        Variance under the null hypothesis
    VA_theta : float
        Variance under the alternative hypothesis
    v1 : float
        Estimated diagnostic performance metric for test 1
    v2 : float
        Estimated diagnostic performance metric for test 2

    Returns
    -------
    int
        Required sample size, rounded up to the nearest integer

    Notes
    -----
    Formula: n = (z_{α/2}·√V_{oθ} + z_β·√V_{Aθ})²/(v₁-v₂)²

    Examples
    --------
    >>> sample_size_for_multireader(alpha=0.05, beta=0.2, Vo_theta=0.04, VA_theta=0.035, v1=0.85, v2=0.75)
    25
    """
    z_alpha = norm.ppf(1 - alpha / 2)
    z_beta = norm.ppf(1 - beta)

    numerator = (z_alpha * math.sqrt(Vo_theta) + z_beta * math.sqrt(VA_theta)) ** 2
    denominator = (v1 - v2) ** 2

    return math.ceil(numerator / denominator)


def noncentrality_parameter_for_multireader_study(J, v1, v2, sigma_b, rho_b, sigma_w, Q, sigma_c, rho_1, rho_2, rho_3):
    """
    Calculate noncentrality parameter for multi-reader studies.

    This function is used for power calculations in complex multi-reader designs
    when detailed statistical modeling of reader effects is needed.

    Parameters
    ----------
    J : int
        Number of readers for each diagnostic test
    v1 : float
        Estimated diagnostic performance metric for test 1
    v2 : float
        Estimated diagnostic performance metric for test 2
    sigma_b : float
        Variability between different readers interpreting same patients using same test
    rho_b : float
        Correlation between readers
    sigma_w : float
        Variability from same reader interpreting same patients using same test at different times
    Q : int
        Number of times each reader interprets each patient's results using same test
    sigma_c : float
        Variability from different samples of patients
    rho_1 : float
        Correlation when same patients evaluated by same reader using different tests
    rho_2 : float
        Correlation when same patients evaluated by different readers using same test
    rho_3 : float
        Correlation when same patients evaluated by different readers using different tests

    Returns
    -------
    float
        Noncentrality parameter (λ)

    Notes
    -----
    Formula: λ = J(v₁-v₂)²/[2(σ_b²(1-ρ_b)+(σ_w²/Q)+σ_c²((1-ρ₁)+(J-1)(ρ₂-ρ₃)))]

    Examples
    --------
    >>> noncentrality_parameter_for_multireader_study(J=5, v1=0.85, v2=0.75, sigma_b=0.02, rho_b=0.6, sigma_w=0.01, Q=1, sigma_c=0.03, rho_1=0.7, rho_2=0.5, rho_3=0.3)
    10.869565217391305
    """
    numerator = J * (v1 - v2) ** 2
    denominator = 2 * (sigma_b ** 2 * (1 - rho_b) +
                       (sigma_w ** 2 / Q) +
                       sigma_c ** 2 * ((1 - rho_1) + (J - 1) * (rho_2 - rho_3)))

    return numerator / denominator


def noncentrality_parameter_for_multireader_multicase_study(v1, v2, J, N, sigma_TR2, sigma_TP2, sigma2):
    """
    Calculate noncentrality parameter for multi-reader multi-case (MRMC) studies.

    This function is used for complex study designs with both reader and case
    variability when comprehensive modeling of variance components is required.

    Parameters
    ----------
    v1 : float
        Estimated diagnostic performance metric for test 1
    v2 : float
        Estimated diagnostic performance metric for test 2
    J : int
        Number of readers
    N : int
        Number of cases
    sigma_TR2 : float
        Variance component related to test-reader interaction
    sigma_TP2 : float
        Variance component related to test-case interaction
    sigma2 : float
        Residual variance

    Returns
    -------
    float
        Noncentrality parameter (λ)

    Notes
    -----
    Formula: λ = (v₁-v₂)²/[(2/(J·N))·(N·σ_{TR}² + J·σ_{TP}² + σ²)]

    Examples
    --------
    >>> noncentrality_parameter_for_multireader_multicase_study(v1=0.85, v2=0.75, J=5, N=100, sigma_TR2=0.01, sigma_TP2=0.02, sigma2=0.03)
    12.5
    """
    numerator = (v1 - v2) ** 2
    denominator = (2 / (J * N)) * (N * sigma_TR2 + J * sigma_TP2 + sigma2)

    return numerator / denominator


def concordance_calculation_Liu(n, r, J, R_0, alpha, beta):
    """
    Calculate sample size for concordance studies using Liu (2005) method.

    This function is used for studies evaluating agreement between diagnostic
    methods or readers when assessing consistency rather than accuracy is the
    primary goal.

    Parameters
    ----------
    n : int
        Initial sample size estimate
    r : float
        Correlation coefficient
    J : int
        Number of readers
    R_0 : float
        Assumed level of concordance
    alpha : float
        Significance level (e.g., 0.05)
    beta : float
        Statistical power (1 - Type II error rate, e.g., 0.8)

    Returns
    -------
    int
        Adjusted sample size needed for concordance study

    Notes
    -----
    Formulas:
        n_{effective} = n × (1 + (J - 1)r)/J
        n_{final} = n_{effective} × ((z_α + z_β)/log((1+R_0)/(1-R_0)))²

    Reference: Liu J, et al. Sample size requirements for agreement studies.

    Examples
    --------
    >>> concordance_calculation_Liu(n=50, r=0.3, J=4, R_0=0.7, alpha=0.05, beta=0.2)
    92
    """
    z_alpha = norm.ppf(1 - alpha)
    z_beta = norm.ppf(1 - beta)

    # Calculate effective sample size
    effective_n = n * (1 + (J - 1) * r) / J

    # Adjustment factor
    adjustment_factor = ((z_alpha + z_beta) / (math.log((1 + R_0) / (1 - R_0)))) ** 2

    # Final sample size
    final_n = effective_n * adjustment_factor

    return math.ceil(final_n)


def sample_size_MRMC_Hillis_Berbaum(alpha, beta, effect_size, readers, cases, sigma_br, sigma_bc, sigma_brc):
    """
    Calculate sample size for MRMC studies using Hillis and Berbaum method.

    This function implements Hillis and Berbaum (2004) method for sample size
    calculation in multi-reader multi-case studies.

    Parameters
    ----------
    alpha : float
        Significance level (e.g., 0.05)
    beta : float
        Statistical power (1 - Type II error rate, e.g., 0.8)
    effect_size : float
        Expected difference in performance metrics
    readers : int
        Number of readers
    cases : int
        Initial estimate of cases
    sigma_br : float
        Reader-related variance component
    sigma_bc : float
        Case-related variance component
    sigma_brc : float
        Reader-case interaction variance component

    Returns
    -------
    int
        Number of cases needed for multi-reader multi-case study

    Notes
    -----
    Formulas:
        Var_{total} = σ_{br}/readers + σ_{bc}/cases + σ_{brc}/(readers·cases)
        n_{cases} = (z_{α/2} + z_β)²·Var_{total}/effect_size²

    Reference: Hillis SL, Berbaum KS. Power estimation for the Dorfman-Berbaum-Metz
    method. Acad Radiol. 2004;11:1260-73.

    Examples
    --------
    >>> sample_size_MRMC_Hillis_Berbaum(alpha=0.05, beta=0.2, effect_size=0.1, readers=5, cases=50, sigma_br=0.02, sigma_bc=0.03, sigma_brc=0.01)
    98
    """
    z_alpha = norm.ppf(1 - alpha / 2)
    z_beta = norm.ppf(1 - beta)

    # Calculate total variance
    total_var = (sigma_br / readers +
                 sigma_bc / cases +
                 sigma_brc / (readers * cases))

    # Calculate required cases
    required_cases = ((z_alpha + z_beta) ** 2 * total_var) / (effect_size ** 2)

    return math.ceil(required_cases)


def two_sided_CI_sample_size(var, alpha, L):
    """
    Calculate sample size for two-sided confidence interval of a diagnostic test parameter.

    This function is suitable for comprehensive accuracy evaluations requiring both
    upper and lower confidence limits. Used when a more complete evaluation of
    diagnostic indicators is needed.

    Parameters
    ----------
    var : float
        Variance of the parameter of interest
    alpha : float
        Significance level (e.g., 0.05)
    L : float
        Precision margin (half-width of the confidence interval)

    Returns
    -------
    int
        Required sample size, rounded up to the nearest integer

    Notes
    -----
    Formula: n = Z_{α/2}² × Var / L²

    Examples
    --------
    >>> two_sided_CI_sample_size(var=0.25, alpha=0.05, L=0.1)
    97
    """
    z_a = norm.ppf(alpha / 2)
    n = z_a ** 2 * var / (L ** 2)
    return math.ceil(n)


def one_sided_CI_high_accuracy_sample_size(alpha, beta, theta, L):
    """
    Calculate sample size for one-sided confidence interval in high accuracy scenarios.

    This function is suitable for scenarios where expected accuracy (sensitivity or
    specificity) is very high, approaching 1. Used for gold standard validation studies
    requiring precise upper confidence limits.

    Parameters
    ----------
    alpha : float
        Significance level (e.g., 0.05)
    beta : float
        Statistical power (1 - Type II error rate, e.g., 0.8)
    theta : float
        Expected sensitivity or specificity value (close to 1)
    L : float
        Precision margin (half-width of the confidence interval)

    Returns
    -------
    int
        Required sample size, rounded up to the nearest integer

    Notes
    -----
    Formula: N = [z_{1-β}√(θ(1-θ)) + z_{1-α}√((θ-L)(1-θ+L))]² / L²

    Based on Flahault (2005) method for high accuracy scenarios.

    Examples
    --------
    >>> one_sided_CI_high_accuracy_sample_size(alpha=0.05, beta=0.2, theta=0.95, L=0.03)
    246
    """
    z_beta = norm.ppf(1 - beta)
    z_alpha = norm.ppf(1 - alpha)

    numerator = (z_beta * math.sqrt(theta * (1 - theta)) +
                 z_alpha * math.sqrt((theta - L) * (1 - theta + L))) ** 2
    denominator = L ** 2

    return math.ceil(numerator / denominator)


def total_sample_for_disease_prevalence(n, prev_p, beta):
    """
    Calculate total required sample size based on disease prevalence.

    This function is used in prospective studies where disease prevalence affects
    required recruitment. It calculates how many total subjects are needed to get
    a specified number of cases with the condition of interest.

    Parameters
    ----------
    n : int
        Required number of subjects with the condition
    prev_p : float
        Prevalence rate of the disease (between 0 and 1)
    beta : float
        Statistical power (1 - Type II error rate, e.g., 0.8)

    Returns
    -------
    int
        Total required sample size, rounded up to the nearest integer

    Notes
    -----
    Formula: (N_total·Prep_p-n)/√(N_total·Prep_p(1-Prep_p))=Z_β

    Examples
    --------
    >>> total_sample_for_disease_prevalence(n=30, prev_p=0.1, beta=0.2)
    355
    """

    def equation(N_total):
        z_b = -norm.ppf(beta)
        N_total = N_total[0]
        return (N_total * prev_p - n) / math.sqrt(N_total * prev_p * (1 - prev_p)) - z_b

    N_total_init = n / prev_p
    N_total_res = fsolve(equation, [N_total_init])
    return math.ceil(N_total_res[0])


def prevalence_based_total_sample_size(n, p, beta):
    """
    Advanced method for calculating total sample size based on disease prevalence.

    This function provides a more sophisticated model for calculating required
    sample size in prospective studies, accounting for disease prevalence.

    Parameters
    ----------
    n : int
        Required number of patients with the condition
    p : float
        Disease prevalence in the population (between 0 and 1)
    beta : float
        Statistical power (1 - Type II error rate, e.g., 0.8)

    Returns
    -------
    int
        Total number of subjects needed in a prospective study

    Notes
    -----
    Formula: N = [2pn - p(1-p)z_β² ± √([p(1-p)z_β² - 2pn]² - 4p²n²)]/2p²

    Examples
    --------
    >>> prevalence_based_total_sample_size(n=40, p=0.15, beta=0.2)
    310
    """
    z_beta = norm.ppf(beta)

    # Coefficients for the quadratic equation
    a = 2 * p ** 2
    b = p * (1 - p) * z_beta ** 2 - 2 * p * n
    c = -p ** 2 * n ** 2

    # Calculate the discriminant
    discriminant = b ** 2 - 4 * a * c
    if discriminant < 0:
        raise ValueError("No valid sample size exists with these parameters")

    # Calculate both solutions
    N1 = (-b + math.sqrt(discriminant)) / (2 * a)
    N2 = (-b - math.sqrt(discriminant)) / (2 * a)

    # Return the smallest positive solution
    if N1 > 0 and N2 > 0:
        return math.ceil(min(N1, N2))
    elif N1 > 0:
        return math.ceil(N1)
    elif N2 > 0:
        return math.ceil(N2)
    else:
        raise ValueError("No positive sample size solution exists")


def clustered_sample_size(n, s, r):
    """
    Adjust sample size for clustered data designs.

    This function accounts for correlation within clusters, adjusting the required
    sample size accordingly. Used for studies with clustered data structures where
    observations within clusters are correlated.

    Parameters
    ----------
    n : int
        Initial uncorrected sample size
    s : int
        Cluster size (number of observations per cluster)
    r : float
        Intraclass correlation coefficient (between 0 and 1)

    Returns
    -------
    float
        Adjusted sample size for clustered design

    Notes
    -----
    Formula: n_c = n × (1 + r × (s - 1))

    Examples
    --------
    >>> clustered_sample_size(n=100, s=5, r=0.1)
    140.0
    """
    nc = n * (1 + r * (s - 1))
    return nc


#
# Area Under Curve (AUC) Analysis
#

def AUC_variance_exponential(A, R):
    """
    Calculate variance of Area Under ROC Curve (AUC) based on exponential distribution.

    This function uses the exponential distribution assumption to calculate AUC
    variance. Used for sample size estimation in diagnostic test evaluation.

    Parameters
    ----------
    A : float
        Area under the ROC curve (AUC) value (between 0.5 and 1)
    R : float
        Sample size ratio between positive and negative cases

    Returns
    -------
    float
        Variance of the AUC estimate

    Notes
    -----
    Formula:
        Q₁ = A/(2-A)
        Q₂ = 2A²/(1+A)
        Var = Q₁/R + Q₂ - A²(1/R + 1)

    Examples
    --------
    >>> AUC_variance_exponential(A=0.8, R=1.0)
    0.04266666666666667
    """
    Q1 = A / (2 - A)
    Q2 = 2 * A ** 2 / (1 + A)
    var = Q1 / R + Q2 - A ** 2 * (1 / R + 1)
    return var


def AUC_variance_binormal(A, R):
    """
    Calculate variance of Area Under ROC Curve (AUC) based on binormal distribution.

    This function uses the binormal distribution assumption to calculate AUC
    variance. Used when underlying variables are assumed to follow a binormal
    distribution.

    Parameters
    ----------
    A : float
        Area under the ROC curve (AUC) value (between 0.5 and 1)
    R : float
        Sample size ratio between negative and positive cases

    Returns
    -------
    float
        Variance of the AUC estimate

    Notes
    -----
    Formula: Var = 0.0099·exp(-a²/2)·(5a² + 8 + (a² + 8)/R)
    where a = 1.414·Φ⁻¹(A)

    Examples
    --------
    >>> AUC_variance_binormal(A=0.8, R=1.0)
    0.035310415221844856
    """
    a = 1.414 * norm.ppf(A)
    var = 0.0099 * math.exp(-a ** 2 / 2) * (5 * a ** 2 + 8 + (a ** 2 + 8) / R)
    return var


def AUC_variance_universal(A):
    """
    Calculate a general-purpose variance of Area Under ROC Curve (AUC).

    This function provides a general method for calculating AUC variance
    applicable to any distribution. Used for robust sample size estimation
    across various diagnostic contexts.

    Parameters
    ----------
    A : float
        Area under the ROC curve (AUC) value (between 0.5 and 1)

    Returns
    -------
    float
        Variance of the AUC estimate

    Notes
    -----
    Formula: Var = A·(1-A)

    This is a simplified formula that can be used when specific distribution
    assumptions cannot be made.

    Examples
    --------
    >>> AUC_variance_universal(A=0.8)
    0.16000000000000003
    """
    var = A * (1 - A)
    return var


def AUC_variance_hanley_mcneil(A, R):
    """
    Calculate variance of AUC using the Hanley and McNeil (1982) method.

    This function calculates the variance of AUC based on the Hanley and McNeil
    approach, which assumes underlying variables follow exponential distribution.

    Parameters
    ----------
    A : float
        Area under the ROC curve (AUC) value (between 0.5 and 1)
    R : float
        Sample size ratio between negative and positive cases

    Returns
    -------
    float
        Variance of the AUC estimate

    Notes
    -----
    Formula: V(Â) = A/(R(2-A)) + 2A²/(1+A) - A²(R+1)/R

    Reference: Hanley JA, McNeil BJ. The meaning and use of the area under
    a receiver operating characteristic (ROC) curve. Radiology. 1982;143:29-36.

    Examples
    --------
    >>> AUC_variance_hanley_mcneil(A=0.8, R=1.0)
    0.04266666666666667
    """
    var = (A / (R * (2 - A)) +
           (2 * A ** 2) / (1 + A) -
           (A ** 2 * (R + 1)) / R)
    return var


def AUC_variance_obuchowski_continuous(A, R):
    """
    Calculate variance of AUC for continuous variables using Obuchowski method.

    This function calculates the variance of AUC for continuous variables based on
    Obuchowski and McClish (1997) method, assuming binormal distribution.

    Parameters
    ----------
    A : float
        Area under the ROC curve (AUC) value (between 0.5 and 1)
    R : float
        Sample size ratio between negative and positive cases

    Returns
    -------
    float
        Variance of the AUC estimate for continuous variables

    Notes
    -----
    Formula: V(Â) = (0.0099×e^(-a²/2))×(5a² + 8 + (a²+8)/R) - (0.0398×a²e^(-a²/2))
    where a = z_A×1.414

    Reference: Obuchowski NA, McClish DK. Sample size determination for diagnostic
    accuracy studies involving binormal ROC curve indices. Stat Med. 1997;16:1529-42.

    Examples
    --------
    >>> AUC_variance_obuchowski_continuous(A=0.8, R=1.0)
    0.035310415221844856
    """
    a = 1.414 * norm.ppf(A)  # z_A × 1.414
    var = ((0.0099 * math.exp(-a ** 2 / 2)) *
           (5 * a ** 2 + 8 + (a ** 2 + 8) / R) -
           (0.0398 * a ** 2 * math.exp(-a ** 2 / 2)))
    return var


def ROC_accuracy_hypothesis_sample_size(alpha, beta, V0, VA, t0, t1):
    """
    Calculate sample size for hypothesis testing of ROC curve accuracy.

    This function calculates the required sample size for testing whether an AUC
    equals a specific value. Used for comparing a diagnostic method with a
    benchmark level.

    Parameters
    ----------
    alpha : float
        Significance level (e.g., 0.05)
    beta : float
        Statistical power (1 - Type II error rate, e.g., 0.8)
    V0 : float
        Variance under the null hypothesis
    VA : float
        Variance under the alternative hypothesis
    t0 : float
        ROC accuracy under the null hypothesis
    t1 : float
        ROC accuracy under the alternative hypothesis

    Returns
    -------
    float
        Required sample size

    Notes
    -----
    Formula: n = (Z_{α/2}√V_0 + Z_β√V_A)² / (t_1 - t_0)²

    Examples
    --------
    >>> ROC_accuracy_hypothesis_sample_size(alpha=0.05, beta=0.2, V0=0.04, VA=0.035, t0=0.7, t1=0.8)
    100.0
    """
    z_a = norm.ppf(alpha / 2)
    z_b = norm.ppf(beta)
    n = (z_a * math.sqrt(V0) + z_b * math.sqrt(VA)) ** 2 / (t1 - t0) ** 2
    return n


#
# ROC Curve Analysis
#

def transformed_sensitivity_at_fixed_FPR(a, b, e):
    """
    Calculate transformed sensitivity at a fixed false positive rate.

    This function calculates the standardized z-transformation of sensitivity
    at a fixed false positive rate. Used for threshold determination and
    optimization.

    Parameters
    ----------
    a : float
        Binormal parameter a
    b : float
        Binormal parameter b
    e : float
        False Positive Rate (FPR) between 0 and 1

    Returns
    -------
    float
        Transformed sensitivity value

    Notes
    -----
    Formula: z = a + b·Φ⁻¹(e)

    Examples
    --------
    >>> transformed_sensitivity_at_fixed_FPR(a=1.5, b=0.8, e=0.1)
    0.22700138999623418
    """
    return a + b * norm.ppf(e)


def variance_transformed_sensitivity(a, b, e, R):
    """
    Calculate variance of transformed sensitivity at fixed false positive rate.

    This function calculates the variance of the transformed sensitivity at a
    fixed false positive rate. Used for confidence interval estimation.

    Parameters
    ----------
    a : float
        Binormal parameter a
    b : float
        Binormal parameter b
    e : float
        False Positive Rate (FPR) between 0 and 1
    R : float
        Sample size ratio between positive and negative cases

    Returns
    -------
    float
        Variance of the transformed sensitivity

    Notes
    -----
    Formula: V(z) = 1 + b²/R + a²/2 + g²b²(1+R)/(2R)
    where g = Φ⁻¹(e)

    Examples
    --------
    >>> variance_transformed_sensitivity(a=1.5, b=0.8, e=0.1, R=1.0)
    2.4617530038240466
    """
    g = norm.ppf(e)
    return (1 + b ** 2 / R + a ** 2 / 2 + g ** 2 * b ** 2 * (1 + R) / (2 * R))


def a_at_fixed_FPR(b, FPR, Se):
    """
    Calculate binormal parameter 'a' at a fixed false positive rate.

    This function calculates the binormal parameter 'a' given parameter 'b',
    the false positive rate, and sensitivity. Used in ROC curve modeling.

    Parameters
    ----------
    b : float
        Empirical parameter b
    FPR : float
        False Positive Rate between 0 and 1
    Se : float
        Sensitivity between 0 and 1

    Returns
    -------
    float
        Calculated a value

    Notes
    -----
    Formula: a = b·Φ⁻¹(1.0-FPR) - Φ⁻¹(1.0-Se)

    Examples
    --------
    >>> a_at_fixed_FPR(b=0.8, FPR=0.1, Se=0.85)
    1.7575365948956835
    """
    # Calculate Φ⁻¹(1.0 - FPR)
    z_fpr = norm.ppf(1.0 - FPR)

    # Calculate Φ⁻¹(1.0 - Se)
    z_sensitivity = norm.ppf(1.0 - Se)

    # Calculate a
    a = b * z_fpr - z_sensitivity

    return a


def compute_f_g(a, b, e1, e2):
    """
    Helper function to compute f and g parameters for partial AUC calculations.

    This function computes intermediate parameters used in partial AUC
    variance calculations.

    Parameters
    ----------
    a : float
        Binormal parameter a
    b : float
        Binormal parameter b
    e1 : float
        Lower bound of FPR range
    e2 : float
        Upper bound of FPR range

    Returns
    -------
    tuple
        (f, g) parameters

    Notes
    -----
    This is a helper function for variance_of_partial_AUC and related functions.
    """
    # Calculate e' and e''
    e1_prime = (norm.ppf(e1) + (a * b) * (1 + b ** 2) ** (-1)) * math.sqrt(1 + b ** 2)
    e2_prime = (norm.ppf(e2) + (a * b) * (1 + b ** 2) ** (-1)) * math.sqrt(1 + b ** 2)

    e1_double_prime = (e1_prime ** 2) / 2
    e2_double_prime = (e2_prime ** 2) / 2

    # Calculate various expressions
    expr1 = math.exp(-a ** 2 / (2 * (1 + b ** 2)))
    expr2 = (1 + b ** 2)
    expr3 = norm.cdf(e2_prime) - norm.cdf(e1_prime)
    expr4 = math.exp(-e1_double_prime) - math.exp(-e2_double_prime)

    # Calculate f
    f = expr1 * (1 / math.sqrt(2 * math.pi * expr2)) * expr3

    # Calculate g
    g = expr1 * (1 / (2 * math.pi * expr2)) * expr4 - (a * b) * expr1 * (2 * math.pi * expr2 ** 3) ** (-0.5) * expr3

    return f, g


def variance_of_partial_AUC(a, b, e1, e2, R):
    """
    Calculate variance of partial Area Under ROC Curve (pAUC).

    This function calculates the variance of partial AUC in a specific false
    positive rate range. Used for studies focusing on specific regions of ROC space.

    Parameters
    ----------
    a : float
        Binormal parameter a
    b : float
        Binormal parameter b
    e1 : float
        Lower bound of the False Positive Rate (FPR)
    e2 : float
        Upper bound of the False Positive Rate (FPR)
    R : float
        Sample ratio (patients vs non-patients)

    Returns
    -------
    float
        Variance of the partial AUC

    Notes
    -----
    Formula: V(A_{e1≤FPR≤e2}) = f²(1 + b²/R + a²/2) + g²[b²(1+R)/(2R)]

    Examples
    --------
    >>> variance_of_partial_AUC(a=1.5, b=0.8, e1=0.01, e2=0.1, R=1.0)
    0.00011691836692750323
    """
    f, g = compute_f_g(a, b, e1, e2)

    # Calculate variance
    variance = (f ** 2 * (1 + b ** 2 / R + a ** 2 / 2) +
                g ** 2 * (b ** 2 * (1 + R) / (2 * R)))

    return variance


def partial_AUC_variance_observed(a, b, e1, e2, R):
    """
    Calculate variance of observed partial AUC for binormal data.

    This function calculates a more complete variance estimation for partial AUC
    that includes covariance terms. Used for more precise statistical analysis of
    ROC curve segments.

    Parameters
    ----------
    a : float
        Binormal parameter a
    b : float
        Binormal parameter b
    e1 : float
        Lower bound of FPR range
    e2 : float
        Upper bound of FPR range
    R : float
        Sample ratio (patients vs non-patients)

    Returns
    -------
    float
        Variance of observed partial AUC

    Notes
    -----
    Formula: V(Â) = f²×(1 + b²/R + a²/2) + g²×[b²(1+R)/(2R)] + f×g×ab

    Examples
    --------
    >>> partial_AUC_variance_observed(a=1.5, b=0.8, e1=0.01, e2=0.1, R=1.0)
    0.00011691836692750323
    """
    f, g = compute_f_g(a, b, e1, e2)

    # Calculate variance with covariance term
    variance = (f ** 2 * (1 + b ** 2 / R + a ** 2 / 2) +
                g ** 2 * (b ** 2 * (1 + R) / (2 * R)) +
                f * g * a * b)

    return variance


#
# Comparison of Diagnostic Methods
#

def sample_size_for_two_diagnostic_tests(alpha, beta, delta, Se1, Se2, coPos):
    """
    Calculate sample size for comparing two diagnostic tests.

    This function calculates the required sample size for directly comparing
    two diagnostic methods' performance, accounting for correlation between tests.
    Used for paired study designs where both tests are applied to the same subjects.

    Parameters
    ----------
    alpha : float
        Significance level (Type I error rate, e.g., 0.05)
    beta : float
        Statistical power (1 - Type II error rate, e.g., 0.8)
    delta : float
        Minimum difference to be detected
    Se1 : float
        Sensitivity of test 1
    Se2 : float
        Sensitivity of test 2
    coPos : float
        P(T1 = 1 | T2 = 1), probability of test 1 being positive given test 2 is positive

    Returns
    -------
    int
        Required sample size, rounded up to the nearest integer

    Notes
    -----
    Formulas:
        ψ = Se₁ + Se₂ - 2·Se₂·P(T1=1|T2=1)
        V₀ = ψ
        V_A = ψ - δ²
        n = (z_{α/2}·√V₀ + z_β·√V_A)² / δ²

    Examples
    --------
    >>> sample_size_for_two_diagnostic_tests(alpha=0.05, beta=0.2, delta=0.1, Se1=0.85, Se2=0.75, coPos=0.6)
    142
    """
    # Calculate ψ (psi)
    psi = Se1 + Se2 - 2 * Se2 * coPos

    # Calculate V_o and V_A
    Vo = psi
    VA = psi - delta ** 2

    # Calculate sample size
    z_alpha = norm.ppf(1 - alpha / 2)
    z_beta = norm.ppf(1 - beta)
    numerator = (z_alpha * math.sqrt(Vo) + z_beta * math.sqrt(VA)) ** 2
    n = numerator / (delta ** 2)

    return math.ceil(n)


def sample_size_for_sensitivity_specificity_comparison(alpha, beta, se1, se2, coPos, delta):
    """
    Calculate sample size for comparing sensitivity or specificity between two tests.

    This function is specialized for focused comparisons of specific accuracy
    parameters. Used when the primary outcome is sensitivity or specificity difference.

    Parameters
    ----------
    alpha : float
        Significance level (Type I error rate, e.g., 0.05)
    beta : float
        Statistical power (1 - Type II error rate, e.g., 0.8)
    se1 : float
        Sensitivity/specificity of test 1
    se2 : float
        Sensitivity/specificity of test 2
    coPos : float
        P(T1 = 1 | T2 = 1), probability of test 1 being positive given test 2 is positive
    delta : float
        Minimum difference to be detected

    Returns
    -------
    int
        Required sample size, rounded up to the nearest integer

    Notes
    -----
    Formulas:
        V₀(θ̂₁-θ̂₂) = se₁ + se₂ - 2×se₂×P(T₁=1|T₂=1)
        V_A(θ̂₁-θ̂₂) = V₀(θ̂₁-θ̂₂) - (se₁-se₂)²
        n = (z_{α/2}·√V₀ + z_β·√V_A)² / δ²

    Examples
    --------
    >>> sample_size_for_sensitivity_specificity_comparison(alpha=0.05, beta=0.2, se1=0.85, se2=0.75, coPos=0.6, delta=0.1)
    142
    """
    z_alpha = norm.ppf(1 - alpha / 2)
    z_beta = norm.ppf(1 - beta)

    # Calculate psi
    psi = se1 + se2 - 2 * se2 * coPos

    # Calculate V_o and V_A
    Vo = psi
    VA = psi - delta ** 2

    # Calculate sample size
    numerator = (z_alpha * math.sqrt(Vo) + z_beta * math.sqrt(VA)) ** 2
    n = numerator / (delta ** 2)

    return math.ceil(n)


def unpaired_sample_size(n1, n):
    """
    Calculate sample size for one test when the sample size of another test is fixed.

    This function is used for unpaired study designs comparing two diagnostic methods,
    when one sample size is constrained by practical considerations.

    Parameters
    ----------
    n1 : int
        Known sample size of the first test
    n : int
        Theoretical sample size computed from formula

    Returns
    -------
    int
        Unpaired sample size (n2), rounded up to the nearest integer

    Notes
    -----
    Formula: n₂ = (n·n₁)/(2n₁-n)

    Examples
    --------
    >>> unpaired_sample_size(n1=50, n=80)
    200
    """
    n2 = (n * n1) / (2 * n1 - n)
    return math.ceil(n2)


def sample_size_rPPV(alpha, beta, gamma, delta, p5, p6, p7, p3, PPV2):
    """
    Calculate sample size for comparing relative Positive Predictive Values.

    This function is used for comparing predictive values rather than
    sensitivity/specificity. Used for studies emphasizing clinical decision-making impact.

    Parameters
    ----------
    alpha : float
        Significance level (Type I error rate, e.g., 0.05)
    beta : float
        Statistical power (1 - Type II error rate, e.g., 0.8)
    gamma : float
        Effect size parameter
    delta : float
        Effect size threshold
    p5 : float
        Proportion of patients testing positive on both tests (for patients with the condition)
    p6 : float
        Proportion testing positive on test 1 and negative on test 2 (for patients with the condition)
    p7 : float
        Proportion testing positive on test 2 and negative on test 1 (for patients with the condition)
    p3 : float
        Proportion testing positive on test 2 and negative on test 1 (for patients without the condition)
    PPV2 : float
        Positive Predictive Value (PPV) for test 2

    Returns
    -------
    int
        Required sample size, rounded up to the nearest integer

    Notes
    -----
    Formula: n = (z_α + z_β)²/lg²(γ/δ)·[(p₅+p₆)(p₅+p₇)]⁻¹·{2(p₇+p₃)γPPV₂² + (-p₆+p₅(1-γ))PPV₂ + p₆+p₇(1-3γPPV₂)}

    Examples
    --------
    >>> sample_size_rPPV(alpha=0.05, beta=0.2, gamma=1.2, delta=1.1, p5=0.4, p6=0.1, p7=0.15, p3=0.05, PPV2=0.8)
    113
    """
    z_alpha = norm.ppf(1 - alpha / 2)
    z_beta = norm.ppf(1 - beta)

    log_term = math.log(gamma / delta, 10)
    term1 = (z_beta + z_alpha) ** 2 / (log_term ** 2)
    term2 = 1 / ((p5 + p6) * (p5 + p7))
    term3 = (2 * (p7 + p3) * gamma * PPV2 ** 2 +
             (-p6 + p5 * (1 - gamma)) * PPV2 +
             p6 + p7 * (1 - 3 * gamma * PPV2))

    n = term1 * term2 * term3
    return math.ceil(n)


def sample_size_rNPV(alpha, beta, gamma, delta, p2, p4, p8, p3, NPV2):
    """
    Calculate sample size for comparing relative Negative Predictive Values.

    This function is used for comparing tests' ability to rule out disease.
    Used for studies focused on exclusion diagnosis capability.

    Parameters
    ----------
    alpha : float
        Significance level (Type I error rate, e.g., 0.05)
    beta : float
        Statistical power (1 - Type II error rate, e.g., 0.8)
    gamma : float
        Effect size parameter
    delta : float
        Effect size threshold
    p2 : float
        Proportion testing positive on test 1 and negative on test 2 (for patients without the condition)
    p4 : float
        Proportion testing negative on both tests (for patients without the condition)
    p8 : float
        Proportion testing negative on both tests (for patients with the condition)
    p3 : float
        Proportion testing positive on test 2 and negative on test 1 (for patients without the condition)
    NPV2 : float
        Negative Predictive Value (NPV) for test 2

    Returns
    -------
    int
        Required sample size, rounded up to the nearest integer

    Notes
    -----
    Formula: n = (z_α + z_β)⁴/lg²(γ/δ)·[(p₂+p₄)(p₃+p₄)]⁻¹·{-2(p₄+p₈)γNPV₂² + (-p₃+p₄-γ(p₂-p₄))NPV₂ + p₂+p₃}

    Examples
    --------
    >>> sample_size_rNPV(alpha=0.05, beta=0.2, gamma=1.2, delta=1.1, p2=0.1, p4=0.4, p8=0.05, p3=0.05, NPV2=0.85)
    156
    """
    z_alpha = norm.ppf(1 - alpha / 2)
    z_beta = norm.ppf(1 - beta)

    log_term = math.log(gamma / delta, 10)
    term1 = (z_beta + z_alpha) ** 4 / (log_term ** 2)
    term2 = 1 / ((p2 + p4) * (p3 + p4))
    term3 = (-2 * (p4 + p8) * gamma * NPV2 ** 2 +
             (-p3 + p4 - gamma * (p2 - p4)) * NPV2 +
             p2 + p3)

    n = term1 * term2 * term3
    return math.ceil(n)


def covariance_for_two_ROC(a1, a2, rD, rN, R):
    """
    Calculate covariance between AUC estimates of two ROC curves.

    This function is used for paired studies comparing overall diagnostic performance
    when correlation between AUC estimates must be accounted for.

    Parameters
    ----------
    a1 : float
        Binormal parameter a for test 1
    a2 : float
        Binormal parameter a for test 2
    rD : float
        Correlation coefficient for diseased cases
    rN : float
        Correlation coefficient for non-diseased cases
    R : float
        Sample size ratio (patients vs non-patients)

    Returns
    -------
    float
        Covariance between AUC of two ROC curves

    Notes
    -----
    Formula: C(A₁,A₂) = e^(-(a₁²+a₂²)/4)/12.5664·(r_D+r_N/R+r_D²a₁a₂/2) + e^(-(a₁²+a₂²)/4)/50.2655·(a₁a₂(r_N²+Rr_D²)/(2R)) - e^(-(a₁²+a₂²)/4)/25.1327·r_D²a₁a₂

    Examples
    --------
    >>> covariance_for_two_ROC(a1=1.5, a2=1.2, rD=0.6, rN=0.5, R=1.0)
    0.045
    """
    term1 = (math.exp(-(a1 ** 2 + a2 ** 2) / 4) / 12.5664 *
             (rD + rN / R + (rD ** 2 * a1 * a2) / 2))
    term2 = (math.exp(-(a1 ** 2 + a2 ** 2) / 4) / 50.2655 *
             ((a1 * a2 * (rN ** 2 + R * rD ** 2)) / (2 * R)))
    term3 = (math.exp(-(a1 ** 2 + a2 ** 2) / 4) / 25.1327 *
             (rD ** 2 * a1 * a2))

    return term1 + term2 - term3


def variance_compare_two_tests(theta1, theta2, r):
    """
    Calculate variances for comparing two diagnostic tests.

    This function calculates variances under null and alternative hypotheses for
    comparing two diagnostic tests. Used for statistical inference in comparative
    diagnostic accuracy studies.

    Parameters
    ----------
    theta1 : float
        Accuracy of test 1
    theta2 : float
        Accuracy of test 2
    r : float
        Correlation coefficient between test 1 and test 2

    Returns
    -------
    tuple
        (V_0, V_A) - Variance under null hypothesis and under alternative hypothesis

    Notes
    -----
    Formulas:
        V₀ = θ₁(1-θ₁) + θ₁(1-θ₁) - 2r√(θ₁(1-θ₁)θ₁(1-θ₁))
        V_A = θ₁(1-θ₁) + θ₂(1-θ₂) - 2r√(θ₁(1-θ₁)θ₂(1-θ₂))

    Examples
    --------
    >>> variance_compare_two_tests(theta1=0.8, theta2=0.75, r=0.6)
    (0.12800000000000003, 0.13000000000000003)
    """
    Vo = (theta1 * (1 - theta1) + theta1 * (1 - theta1) -
          2 * r * math.sqrt(theta1 * (1 - theta1) * theta1 * (1 - theta1)))

    VA = (theta1 * (1 - theta1) + theta2 * (1 - theta2) -
          2 * r * math.sqrt(theta1 * (1 - theta1) * theta2 * (1 - theta2)))

    return Vo, VA


def covariance_compare_sensitivity_fixed_FPR(a1, a2, b1, b2, rD, rN, R, e):
    """
    Calculate covariance between sensitivity estimates at fixed false positive rate.

    This function is used for paired-design studies comparing test performance at
    specific thresholds when correlation between tests must be accounted for at fixed
    operating points.

    Parameters
    ----------
    a1 : float
        Binormal parameter a for test 1
    a2 : float
        Binormal parameter a for test 2
    b1 : float
        Binormal parameter b for test 1
    b2 : float
        Binormal parameter b for test 2
    rD : float
        Correlation coefficient for diseased group
    rN : float
        Correlation coefficient for non-diseased group
    R : float
        Ratio of non-diseased to diseased samples
    e : float
        False Positive Rate (FPR)

    Returns
    -------
    float
        Covariance between sensitivity estimates at fixed FPR

    Notes
    -----
    Formula: C = r_D + r_N·b₁b₂/R + r_D²a₁a₂/2 + g²b₁b₂(r_N²+Rr_D²)/(2R) + gr_D²/2·(a₁b₂+a₂b₁)
    where g = Φ⁻¹(e)

    Examples
    --------
    >>> covariance_compare_sensitivity_fixed_FPR(a1=1.5, a2=1.2, b1=0.8, b2=0.9, rD=0.6, rN=0.5, R=1.0, e=0.1)
    0.6636
    """
    g = norm.ppf(e)

    term1 = rD + (rN * b1 * b2) / R + (rD ** 2 * a1 * a2) / 2
    term2 = (g ** 2 * b1 * b2 * (rN ** 2 + R * rD ** 2)) / (2 * R)
    term3 = (g * rD ** 2 / 2) * (a1 * b2 + a2 * b1)

    return term1 + term2 + term3


def covariance_partial_AUC(a1, a2, b1, b2, e1, e2, rD, rN, R):
    """
    Calculate covariance between partial AUC estimates for two diagnostic tests.

    This function is used for comparing test performance in specific ROC curve regions
    when correlation between partial AUC estimates must be accounted for.

    Parameters
    ----------
    a1 : float
        Binormal parameter a for test 1
    a2 : float
        Binormal parameter a for test 2
    b1 : float
        Binormal parameter b for test 1
    b2 : float
        Binormal parameter b for test 2
    e1 : float
        Lower bound of the False Positive Rate (FPR)
    e2 : float
        Upper bound of the False Positive Rate (FPR)
    rD : float
        Correlation coefficient for diseased group
    rN : float
        Correlation coefficient for non-diseased group
    R : float
        Ratio of non-diseased to diseased samples

    Returns
    -------
    float
        Covariance between partial AUC estimates

    Notes
    -----
    Formula: C = f₁f₂(r_D + r_Nb₁b₂/R + r_D²a₁a₂/2) + g₁g₂[b₁b₂(r_N²+Rr_D²)/(2R)] + f₁g₂·r_D²a₁b₂/2 + f₂g₁·r_D²a₂b₁

    Examples
    --------
    >>> covariance_partial_AUC(a1=1.5, a2=1.2, b1=0.8, b2=0.9, e1=0.01, e2=0.1, rD=0.6, rN=0.5, R=1.0)
    0.0002
    """
    f1, g1 = compute_f_g(a1, b1, e1, e2)
    f2, g2 = compute_f_g(a2, b2, e1, e2)

    term1 = f1 * f2 * (rD + (rN * b1 * b2) / R + (rD ** 2 * a1 * a2) / 2)
    term2 = g1 * g2 * (b1 * b2 * (rN ** 2 + R * rD ** 2)) / (2 * R)
    term3 = f1 * g2 * (rD ** 2 * a1 * b2 / 2) + f2 * g1 * (rD ** 2 * a2 * b1)

    return term1 + term2 + term3


def relative_TPR_FPR(TPR1, TPR2, FPR1, FPR2):
    """
    Calculate relative True Positive Rate (rTPR) and relative False Positive Rate (rFPR).

    This function is used for expressing diagnostic accuracy comparisons as ratios
    when relative performance metrics are more meaningful than absolute differences.

    Parameters
    ----------
    TPR1 : float
        True Positive Rate of test 1
    TPR2 : float
        True Positive Rate of test 2
    FPR1 : float
        False Positive Rate of test 1
    FPR2 : float
        False Positive Rate of test 2

    Returns
    -------
    tuple
        (rTPR, rFPR) - Relative True Positive Rate and Relative False Positive Rate

    Notes
    -----
    Formulas:
        rTPR = TPR₁/TPR₂
        rFPR = FPR₁/FPR₂

    Examples
    --------
    >>> relative_TPR_FPR(TPR1=0.85, TPR2=0.75, FPR1=0.12, FPR2=0.1)
    (1.1333333333333333, 1.2)
    """
    rTPR = TPR1 / TPR2
    rFPR = FPR1 / FPR2
    return rTPR, rFPR


#
# Non-inferiority and Equivalence Testing
#

def non_inferiority_sample_size(alpha, beta, theta_S, theta_E, delta_M, var):
    """
    Calculate sample size for non-inferiority testing of diagnostic methods.

    This function is used for demonstrating that a new test is not inferior to a
    standard test, especially when the new method has other advantages like being
    less expensive or invasive.

    Parameters
    ----------
    alpha : float
        Significance level (Type I error rate, e.g., 0.05)
    beta : float
        Statistical power (1 - Type II error rate, e.g., 0.8)
    theta_S : float
        Standard test parameter
    theta_E : float
        Experimental test parameter
    delta_M : float
        Non-inferiority margin
    var : float
        Variance of the estimate

    Returns
    -------
    int
        Required sample size, rounded up to the nearest integer

    Notes
    -----
    Formula: n = (Z_α + Z_β)²·Var/(θ_S - θ_E - δ_M)²

    Examples
    --------
    >>> non_inferiority_sample_size(alpha=0.05, beta=0.2, theta_S=0.85, theta_E=0.8, delta_M=0.1, var=0.04)
    27
    """
    z_alpha = norm.ppf(1 - alpha)
    z_beta = norm.ppf(1 - beta)

    numerator = (z_alpha + z_beta) ** 2 * var
    denominator = (theta_S - theta_E - delta_M) ** 2

    n = numerator / denominator
    return math.ceil(n)


def equivalence_sample_size(alpha, beta, theta_S, theta_E, delta_L, delta_U, var):
    """
    Calculate sample size for equivalence testing between diagnostic methods.

    This function is used when the goal is to prove two methods are statistically
    equivalent, rather than focusing on superiority or non-inferiority.

    Parameters
    ----------
    alpha : float
        Significance level (e.g., 0.05)
    beta : float
        Statistical power (1 - Type II error rate, e.g., 0.8)
    theta_S : float
        Standard treatment effect
    theta_E : float
        Experimental treatment effect
    delta_L : float
        Lower equivalence margin
    delta_U : float
        Upper equivalence margin
    var : float
        Variance of the effect size

    Returns
    -------
    int
        Required sample size, rounded up to the nearest integer

    Notes
    -----
    Formula depends on (θ_S - θ_E):
        If > 0: n = (Z_α + Z_β)²·var/(δ_U - (θ_S - θ_E))²
        If < 0: n = (Z_α + Z_β)²·var/(δ_U + (θ_S - θ_E))²
        If = 0: n = (Z_α + Z_β)²·var/(δ_U²/4)

    Examples
    --------
    >>> equivalence_sample_size(alpha=0.05, beta=0.2, theta_S=0.85, theta_E=0.82, delta_L=-0.1, delta_U=0.1, var=0.04)
    61
    """
    z_alpha = norm.ppf(1 - alpha)
    z_beta = norm.ppf(1 - beta)
    diff = theta_S - theta_E

    if diff > 0:
        denominator = (delta_U - diff) ** 2
    elif diff < 0:
        denominator = (delta_U + diff) ** 2
    else:
        denominator = delta_U ** 2 / 4

    numerator = (z_alpha + z_beta) ** 2 * var
    n = numerator / denominator
    return math.ceil(n)