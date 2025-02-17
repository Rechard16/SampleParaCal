
import math
from scipy.optimize import fsolve
from scipy.stats import norm


def one_sided_ci_sample_size(alpha, beta, var, L):#6.1
    z_a = norm.ppf(alpha / 2)
    z_b = norm.ppf(beta)
    n = (z_a + z_b) ** 2 * var / (L ** 2)
    return math.ceil(n)

def two_sided_ci_sample_size(var,alpha,L):#6.2
    z_a=norm.ppf(alpha/2)
    n=z_a**2*var/(L**2)
    return math.ceil(n)
def total_sample_for_disease_prevalence(n,prev_p,beta):#6.3
    def equation(N_total):
        z_b=-norm.ppf(beta)
        N_total=N_total[0]
        return (N_total*prev_p-n)/math.sqrt(N_total * prev_p * (1 - prev_p)) - z_b
    N_total_init=n/prev_p
    N_total_res=fsolve(equation,[N_total_init])
    return math.ceil(N_total_res[0])
def roc_variance_exponential(A,R):#6.4
    Q1=A/(2-A)
    Q2=2*A**2/(1+A)
    var=Q1/R+Q2-A**2*(1/R+1)
    return var
def roc_variance_binormal(A,R):#6.5
    a=1.414*norm.ppf(A)
    var=0.0099*math.exp(-a**2/2)*(5*a**2+8+(a**2+8)/R)
    return var
def roc_variance_universal(A):#6.6
    var=A*(1-A)
    return var
def clustered_sample_size(n,s,r):#6.7
    nc=n*(1+r*(s-1))
    return nc
def roc_accuracy_hypothesis_sample_size(alpha,beta,V0,VA,t0,t1):#6.8
    z_a=norm.ppf(alpha/2)
    z_b=norm.ppf(beta)
    n=(z_a*math.sqrt(V0)+z_b*math.sqrt(VA))**2/(t1-t0)**2
    return n
def transformed_sensitivity_at_fixed_fpr(a, b, e):#6.9
    """计算灵敏度的标准化 z 变换 (公式 6.9)"""
    return a + b * norm.ppf(e)
def variance_transformed_sensitivity(a, b, e, R):#6.10
    """计算 V(z(Sep_{FPR=f})) 的方差 (公式 6.10)"""
    g=norm.ppf(e)
    return 1 + b**2/R +a ** 2 /2+ g**2 * b ** 2 * (1 + R) / (2 * R)
def transformed_sensitivity_at_fixed_fpr(b, FPR, Se):#6.11
    """
    计算公式 (6.11) 左侧的 a 值

    :param b: 经验参数 b
    :param FPR: 假阳性率 (False Positive Rate)
    :param Se: 敏感度 (Sensitivity)
    :return: 计算得到的 a 值
    """
    # 计算 Φ⁻¹(1.0 - FPR)
    z_fpr = norm.ppf(1.0 - FPR)

    # 计算 Φ⁻¹(1.0 - Se)
    z_sensitivity = norm.ppf(1.0 - Se)

    # 计算 a
    a = b * z_fpr - z_sensitivity

    return a

def variance_of_partial_roc_area(a, b, e1, e2, R):#6.12
    """
    计算部分 ROC 曲线面积 (pAUC) 在 FPR 范围 e1 到 e2 内的方差 (公式 6.12)

    :param a: binormal 参数 a
    :param b: binormal 参数 b
    :param e1: FPR 下界
    :param e2: FPR 上界
    :param R: 样本比例 (患者 vs 非患者)
    :return: 部分 AUC 的方差
    """

    # 计算 e' 和 e''
    e1_prime = (norm.ppf(e1) + (a * b) * (1 + b ** 2) ** (-1)) * math.sqrt(1 + b ** 2)
    e2_prime = (norm.ppf(e2) + (a * b) * (1 + b ** 2) ** (-1)) * math.sqrt(1 + b ** 2)

    e1_double_prime = (e1_prime ** 2) / 2
    e2_double_prime = (e2_prime ** 2) / 2

    # 计算各个 expr
    expr1 = math.exp(-a ** 2 / (2 * (1 + b ** 2)))
    expr2 = (1 + b ** 2)
    expr3 = norm.cdf(e2_prime) - norm.cdf(e1_prime)
    expr4 = math.exp(-e1_double_prime) - math.exp(-e2_double_prime)

    # 计算 f
    f = expr1 * (1 / math.sqrt(2 * math.pi * expr2)) * expr3
    print(f)
    # 计算 g
    g = expr1 * (1 / (2 * math.pi * expr2)) * expr4 - (a * b) * expr1 * ( 2 * math.pi * expr2 ** 3)**(-0.5) * expr3
    print(g)
    # 计算 V(A_{e1 <= FPR <= e2})
    variance = f ** 2 * (1 + b ** 2 / R + a ** 2 / 2) + g ** 2 * (b ** 2 * (1 + R) / (2 * R))

    return variance


def sample_size_for_two_diagnostic_tests(alpha, beta, delta, Se1, Se2, coPos):
    """
    计算诊断测试样本量 (公式 6.13 - 6.16)

    :param alpha: 显著性水平 (Type I error rate)
    :param beta: 统计效能 (Type II error rate, 1 - Power)
    :param theta1: 诊断测试 1 的准确度 (θ1)
    :param theta2: 诊断测试 2 的准确度 (θ2)
    :param Se1: 备择假设下测试 1 的灵敏度
    :param Se2: 备择假设下测试 2 的灵敏度
    :param copos: P(T1 = 1 | T2 = 1)，测试 1 结果为阳性的概率 (给定测试 2 结果为阳性)
    :return: 样本量 n
    """
    # 计算 ψ (公式 6.16)
    psi = Se1 + Se2 - 2 * Se2 * coPos

    # 计算 V_o 和 V_A (公式 6.15)

    Vo = psi
    VA = psi - delta ** 2

    # 计算样本量 (公式 6.13)
    z_alpha = norm.ppf(1 - alpha / 2)
    z_beta = norm.ppf(1 - beta)
    numerator = (z_alpha * math.sqrt(Vo) + z_beta * math.sqrt(VA)) ** 2
    n = numerator / (delta ** 2)

    return math.ceil(n)


def unpaired_sample_size(n1, n):
    """
    计算非配对研究中的未知样本量 n2 (公式 6.17)

    :param n1: 已知测试的固定样本量
    :param n: 由公式 (6.13) 计算出的理论样本量
    :return: 计算得到的未知测试样本量 n2
    """
    return math.ceil((n * n1) / (2 * n1 - n))

def sample_size_rPPV(alpha, beta, gamma, delta, p5, p6, p7, p3, PPV2):
    """
    计算 rPPV 相关的样本量 (公式 6.18)
    """
    z_alpha = norm.ppf(1 - alpha / 2)
    z_beta = norm.ppf(1 - beta)
    log_term = math.log(gamma / delta,math.e)
    term1 = (z_beta + z_alpha) ** 2 / (log_term ** 2)
    term2 = 1 / ((p5 + p6) * (p5 + p7))
    term3 = 2 * (p7 + p3) * gamma * PPV2 ** 2 + (-p6 + p5 * (1 - gamma)) * PPV2 + p6 + p7 * (1 - 3 * gamma * PPV2)
    n = term1 * term2 * term3
    return math.ceil(n)

def sample_size_rNPV(alpha, beta, gamma, delta, p2, p4, p8, p3, NPV2):
    """
    计算 rNPV 相关的样本量 (公式 6.19)
    """
    z_alpha = norm.ppf(1 - alpha / 2)
    z_beta = norm.ppf(1 - beta)

    log_term = math.log(gamma / delta)
    print(log_term)
    term1 = (z_beta + z_alpha) ** 4 / (log_term ** 2)
    term2 = 1 / ((p2 + p4) * (p3 + p4))
    term3 = -2 * (p4 + p8) * gamma * NPV2 ** 2 + (-p3 + p4 - gamma * (p2 - p4)) * NPV2 + p2 + p3
    n = term1 * term2 * term3
    return math.ceil(n)

def covariance_for_two_roc(a1, a2, rD, rN, R):
    """
    计算 Ĉ(Â1, Â2) (公式 6.20)
    """
    term1 = math.exp(-(a1**2 + a2**2) / 4) / 12.5664 * (rD + rN / R + (rD**2 * a1 * a2) / 2)
    term2 = math.exp(-(a1**2 + a2**2) / 4) / 50.2655 * ((a1 * a2 * (rN**2 + R * rD**2)) / (2 * R))
    term3 = math.exp(-(a1**2 + a2**2) / 4) / 25.1327 * (rD**2 * a1 * a2)
    return term1 + term2 - term3
def variance_compare_two_tests(theta1, theta2, r):
    """
    计算 V_o 和 V_A (公式 6.21)
    """
    Vo = theta1 * (1 - theta1) + theta1 * (1 - theta1) - 2 * r * math.sqrt(theta1 * (1 - theta1) * theta1 * (1 - theta1))
    VA = theta1 * (1 - theta1) + theta2 * (1 - theta2) - 2 * r * math.sqrt(theta1 * (1 - theta1) * theta2 * (1 - theta2))
    return Vo, VA

def covariance_compare_sensitivity_fixed_fpr(a1, a2, b1, b2, rD, rN, R, e):
    """
    计算 Ĉ((Ŝ_{e_{FPR=e}})₁, (Ŝ_{e_{FPR=e}})₂) (公式 6.22)
    """
    g = norm.ppf(e)
    term1 = rD + (rN * b1 * b2) / R + (rD**2 * a1 * a2) / 2
    term2 = (g**2 * b1 * b2 * (rN**2 + R * rD**2)) / (2 * R)
    term3 = (g * rD**2 / 2) * (a1 * b2 + a2 * b1)
    return term1 + term2 + term3


def compute_f_g(a, b, e1, e2):
    """
    计算 f 和 g (参考公式 6.12)
    """

    e1_prime = (norm.ppf(e1) + (a * b) * (1 + b ** 2) ** (-1)) * math.sqrt(1 + b ** 2)
    e2_prime = (norm.ppf(e2) + (a * b) * (1 + b ** 2) ** (-1)) * math.sqrt(1 + b ** 2)

    e1_double_prime = (e1_prime ** 2) / 2
    e2_double_prime = (e2_prime ** 2) / 2

    # 计算各个 expr
    expr1 = math.exp(-a ** 2 / (2 * (1 + b ** 2)))
    expr2 = (1 + b ** 2)
    expr3 = norm.cdf(e2_prime) - norm.cdf(e1_prime)
    expr4 = math.exp(-e1_double_prime) - math.exp(-e2_double_prime)

    # 计算 f
    f = expr1 * (1 / math.sqrt(2 * math.pi * expr2)) * expr3
    print(f)
    # 计算 g
    g = expr1 * (1 / (2 * math.pi * expr2)) * expr4 - (a * b) * expr1 * (2 * math.pi * expr2 ** 3) ** (-0.5) * expr3
    return f, g


def covariance_partial_AUC(a1, a2, b1, b2, e1, e2, rD, rN, R):
    """
    计算 Ĉ((Â_{e₁≤FPR≤e₂})₁, (Â_{e₁≤FPR≤e₂})₂) (公式 6.23)
    """
    f1, g1 = compute_f_g(a1, b1, e1, e2)
    f2, g2 = compute_f_g(a2, b2, e1, e2)

    term1 = f1 * f2 * (rD + (rN * b1 * b2) / R + (rD ** 2 * a1 * a2) / 2)
    term2 = g1 * g2 * (b1 * b2 * (rN ** 2 + R * rD ** 2)) / (2 * R)
    term3 = f1 * g2 * (rD ** 2 * a1 * b2 / 2) + f1 * g2 * (rD ** 2 * a2 * b1)

    return term1 + term2 + term3

def non_inferiority_sample_size(alpha, beta, theta_S, theta_E, delta_M, var):
    """
    计算非劣效性检验的样本量 (公式 6.24)
    """
    z_alpha = norm.ppf(1 - alpha)
    z_beta = norm.ppf(1 - beta)
    numerator = (z_alpha + z_beta) ** 2 * var
    denominator = (theta_S - theta_E - delta_M) ** 2
    n = numerator / denominator
    return math.ceil(n)


def equivalence_sample_size(alpha, beta, theta_S, theta_E, delta_L, delta_U, var):
    """
    计算等效性检验的样本量 (公式 6.25 - 6.27)
    """
    z_alpha = norm.ppf(1 - alpha)
    z_beta = norm.ppf(1 - beta)
    diff = theta_S - theta_E

    if diff > 0:
        denominator = (delta_U - diff) ** 2
    elif diff < 0:
        denominator = (delta_U + diff) ** 2
    else:
        z_beta /= 2
        denominator = delta_U ** 2

    numerator = (z_alpha + z_beta) ** 2 * var
    n = numerator / denominator
    return math.ceil(n)

def relative_tpr_fpr(TPR1, TPR2, FPR1, FPR2):
    """
    计算相对真阳性率 (rTPR) 和相对假阳性率 (rFPR) (公式 6.28)
    """
    rTPR = TPR1 / TPR2
    rFPR = FPR1 / FPR2
    return rTPR, rFPR
def rTPR_sample_size(alpha, beta, gamma, delta_1, TPR1, TPR2, TPPR):
    """
    计算 rTPR 相关的样本量 (公式 6.29)
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
    计算 Schafer (1989) 方法的样本量上界 (公式 6.30)
    """
    lambda_val = norm.ppf(SE_SP) - norm.ppf(SE_prime)
    vx = b * math.sqrt(1 + 0.5 * norm.ppf(SP_prime) ** 2)
    vy = math.sqrt(1 + 0.5 * norm.ppf(SE_prime) ** 2)

    numerator = (math.sqrt(2) * norm.ppf(math.sqrt(1 - alpha)) + norm.ppf(1 - beta)) ** 2 * (vx + vy) ** 2
    denominator = lambda_val ** 2

    N = numerator / denominator
    return math.ceil(N)
def noncentrality_parameter_for_multireader_study(J, v1, v2, sigma_b, rho_b, sigma_w, Q, sigma_c, rho_1, rho_2, rho_3):
    """
    计算 λ (公式 6.32)
    """
    numerator = J * (v1 - v2) ** 2
    denominator = 2 * (sigma_b**2 * (1 - rho_b) + (sigma_w**2 / Q) + sigma_c**2 * ((1 - rho_1) + (J - 1) * (rho_2 - rho_3)))
    return numerator / denominator
def variance_for_multireader(Vo_theta, J, rho_dr):
    """
    计算固定读者 MRMC 设计的方差函数 (公式 6.33)
    """
    return Vo_theta * (1 / J + (J - 1) * rho_dr / J)

def sample_size_for_multireader(alpha, beta, Vo_theta, VA_theta, v1, v2):
    """
    计算固定读者 MRMC 设计的样本量 (公式 6.34)
    """
    z_alpha = norm.ppf(1 - alpha / 2)
    z_beta = norm.ppf(1 - beta)
    numerator = (z_alpha * math.sqrt(Vo_theta) + z_beta * math.sqrt(VA_theta)) ** 2
    denominator = (v1 - v2) ** 2
    return math.ceil(numerator / denominator)
def noncentrality_parameter_for_multireader_multicase_study(v1, v2, J, N, sigma_TR2, sigma_TP2, sigma2):
    """
    计算非中心性参数 λ (公式 6.35)
    """
    numerator = (v1 - v2) ** 2
    denominator = (2 / (J * N)) * (N * sigma_TR2 + J * sigma_TP2 + sigma2)
    return numerator / denominator

