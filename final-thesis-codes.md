## Classical SABR Script

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sigma_0 = 0.2
α_values = [0.1, 0.2, 0.3, 0.4]
T_values = [0.5, 1.0, 2.0, 5.0]
M = 10_000
N = 252

results = []

for T in T_values:
    dt = T / N
    time_grid = np.linspace(dt, T, N)

    for α in α_values:
        A_values = np.zeros(M)

        for i in range(M):
            Z = np.random.normal(0, 1, N)
            Z_cumsum = np.cumsum(Z) * np.sqrt(dt)
            sigma_path = sigma_0 * np.exp(α * Z_cumsum - 0.5 * α**2 * time_grid)
            A_values[i] = np.sum(sigma_path**2) * dt

        E_sqrt_A = np.mean(np.sqrt(A_values))
        sqrt_E_A = np.sqrt(np.mean(A_values))
        adjustment = E_sqrt_A - sqrt_E_A

        results.append({
            'T': T,
            'α': α,
            'E[sqrt(A)]': E_sqrt_A,
            'sqrt(E[A])': sqrt_E_A,
            'Convexity Adjustment': adjustment
        })

df_results = pd.DataFrame(results).round(6)
print(df_results)

sns.set_style("white")
plt.figure(figsize=(10, 6))

for α in α_values:
    subset = df_results[df_results['α'] == α]
    plt.plot(
        subset['T'],
        subset['Convexity Adjustment'],
        linestyle='-',
        linewidth=2.0,
        label=f'α = {α:.1f}'
    )

plt.grid(False)
plt.title("Convexity Adjustment under SABR", fontsize=14)
plt.xlabel("T (years)", fontsize=12)
plt.ylabel("Convexity Adjustment", fontsize=12)

plt.legend(title="Alpha Values", loc="lower left")
plt.tight_layout()
plt.savefig("convexity_adjustment.png", dpi=300, bbox_inches="tight")
plt.show()


```

## Calibration Script
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit

# PARAMETERS
sigma_0 = 0.2                 
α_values = [0.1, 0.2, 0.3, 0.4]   
T_values = [0.1,0.3, 0.5, 0.7, 1.0]   
M = 10000                          
N = 252                           

results = []

# MONTE CARLO
for T in T_values:
    for α in α_values:
        dt = T / N
        time_grid = np.linspace(dt, T, N)
        A_values = np.zeros(M)

        for i in range(M):
            Z = np.random.normal(0, 1, N)
            Z_cumsum = np.cumsum(Z) * np.sqrt(dt)
            sigma_path = sigma_0 * np.exp(α * Z_cumsum - 0.5 * α**2 * time_grid)
            A_values[i] = np.sum(sigma_path**2) * dt

        
        E_sqrt_A = np.mean(np.sqrt(A_values))
        sqrt_E_A = np.sqrt(np.mean(A_values))
        adjustment = E_sqrt_A - sqrt_E_A

        results.append({
            'T': T,
            'α': α,
            'E[sqrt(A)]': E_sqrt_A,
            'sqrt(E[A])': sqrt_E_A,
            'Convexity Adjustment': adjustment
        })

# RESULTS TABLE
df_results = pd.DataFrame(results)
print(df_results.round(6))

# CALIBRATION

def negative_power_law(T, c, α):
    return -c * T**α

for alpha_value in [0.1, 0.2, 0.3, 0.4]:
    
    subset = df_results[df_results['α'] == alpha_value]
    T_data = subset['T'].values
    adjustment_data = subset['Convexity Adjustment'].values 
    
    popt, _ = curve_fit(negative_power_law, T_data, adjustment_data)
    c_fit, α_fit = popt
    
    print(f"For α = {alpha_value}, fitted function: "
          f"f(T) = -{c_fit:.5f} * T^({α_fit:.5f})")

    
    T_fit = np.linspace(min(T_data), max(T_data), 100)

   
    plt.scatter(T_data, adjustment_data, label=f"Data (α={alpha_value})")
    plt.plot(T_fit, negative_power_law(T_fit, c_fit, α_fit),
             linestyle='--', label=f"Fit (α={alpha_value})")

plt.legend()
plt.xlabel("T(years)")
plt.ylabel("Convexity Adjustment")
plt.title("Calibration of Convexity Adjustment")
plt.show()

```

## Fractional Brownian Motion and Cholesky  decomposition (under Rough SABR) Script
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.linalg import cholesky
from scipy.optimize import curve_fit

np.random.seed(42)
sigma_0 = 0.20
alpha = 0.20
H_values = [0.3, 0.5, 0.7]
T_values = [0.5, 1.0, 2.0, 5.0]
M, N = 10_000, 252

def fbm_paths_cholesky(H, T, N, M):
    dt = T / N
    t_grid = np.linspace(dt, T, N)
    t_pow = t_grid ** (2 * H)
    cov = 0.5 * (t_pow[:, None] + t_pow[None, :] -
                 np.abs(t_grid[:, None] - t_grid[None, :]) ** (2 * H))
    L = cholesky(cov, lower=True)
    Z = np.random.normal(size=(M, N))
    return t_grid, Z @ L.T

rows = []
for H in H_values:
    for T in T_values:
        t, fbm = fbm_paths_cholesky(H, T, N, M)
        dt = T / N
        sigma_path = sigma_0 * np.exp(alpha * np.sqrt(2 * H) * fbm -
                                      0.5 * alpha**2 * t**(2 * H))
        A = np.sum(sigma_path**2, axis=1) * dt / T
        conv_adjust = np.mean(np.sqrt(A)) - np.sqrt(np.mean(A))
        rows.append({
            "H": H,
            "T": T,
            "Convexity Adjustment": conv_adjust
        })

df = pd.DataFrame(rows)

def neg_power(T, c, beta):
    return -c * (T ** beta)

sns.set_style("white")
plt.figure(figsize=(8, 5))

for idx, H in enumerate(H_values):
    data = df[df["H"] == H]
    popt, _ = curve_fit(neg_power, data["T"], data["Convexity Adjustment"])
    c_fit, beta_fit = popt
    print(f"H = {H:.1f}:  f(T) ≈ -{c_fit:.6f} · T^{beta_fit:.6f}")

    T_fit = np.linspace(min(T_values), max(T_values), 200)
    fitted_curve = neg_power(T_fit, *popt)

    plt.plot(
        T_fit,
        fitted_curve,
        linestyle='-',
        linewidth=2.0,
        color=f"C{idx}",
        label=f"H = {H:.1f}"
    )

plt.grid(False)
plt.title("Convexity Adjustment under fBm", fontsize=14)
plt.xlabel("T (years)", fontsize=12)
plt.ylabel("Convexity Adjustment", fontsize=12)
plt.legend(title="H values", loc="lower left")
plt.tight_layout()
plt.show()


```


