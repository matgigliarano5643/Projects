import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold, cross_val_score

def simulate_asian_options():
    print("Avvio simulazione Monte Carlo per Opzioni Asiatiche (N = 100.000 percorsi)...")
    
    # Parametri finanziari dal documento HW3
    S0 = 100.0
    r = 0.01
    sigma = 0.20
    TT = 1.0
    m = 3
    dt = TT / m
    K = 100.0
    n_paths = 100000
    
    np.random.seed(27)
    
    # Matrice dei movimenti browniani normali N(0,1)
    M = np.random.normal(0, 1, size=(n_paths, m))
    S = np.zeros((n_paths, m))
    
    drift = (r - 0.5 * sigma**2) * dt
    diffusion = sigma * np.sqrt(dt)
    
    # Evoluzione temporale del sottostante nei 3 monitoraggi quadrimestrali
    S[:, 0] = S0 * np.exp(drift + diffusion * M[:, 0])
    for j in range(1, m):
        S[:, j] = S[:, j-1] * np.exp(drift + diffusion * M[:, j])
        
    # Calcolo della media aritmetica A lungo il percorso e del prezzo a scadenza S_T
    A = np.mean(S, axis=1)
    St = S[:, -1]
    
    # Payoff delle 4 tipologie di opzioni asiatiche
    payoffs = {
        'Call_Avg_Price': np.maximum(A - K, 0),
        'Put_Avg_Price': np.maximum(K - A, 0),
        'Put_Avg_Strike': np.maximum(A - St, 0),
        'Call_Avg_Strike': np.maximum(St - A, 0)
    }
    
    discount_factor = np.exp(-r * TT)
    prices = {k: discount_factor * np.mean(v) for k, v in payoffs.items()}
    
    print("\n--- Prezzi delle Opzioni Asiatiche (Risk-Neutral Monte Carlo) ---")
    for k, v in prices.items():
        print(f"  {k:18s}: {v:.4f} €")
        
    # Creazione del DataFrame per il Machine Learning
    df_paths = pd.DataFrame(S, columns=[f'S_t{i+1}' for i in range(m)])
    df_paths['A_mean'] = A
    df_paths['S_T'] = St
    df_paths['path_std'] = np.std(S, axis=1)
    df_paths['target_payoff'] = payoffs['Call_Avg_Price']
    
    return df_paths, S, A, St, payoffs


def run_ml_surrogate_model(df_paths: pd.DataFrame):
    print("\nAddestramento Pipeline ML Surrogata (Pricing Prediction)...")
    
    # Campioniamo 20.000 percorsi per efficienza computazionale
    sample_df = df_paths.sample(n=20000, random_state=42)
    
    X = sample_df[['S_t1', 'S_t2', 'S_t3', 'A_mean', 'S_T', 'path_std']].values
    y = sample_df['target_payoff'].values
    
    # Pipeline ufficiale: Scaler -> PCA -> Gradient Boosting Regressor
    ml_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=3, random_state=42)),
        ('regressor', GradientBoostingRegressor(
            n_estimators=100, 
            max_depth=4, 
            learning_rate=0.1, 
            random_state=42
        ))
    ])
    
    # Validazione incrociata a 5 fold (KFold)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_r2 = cross_val_score(ml_pipeline, X, y, cv=kf, scoring='r2')
    cv_rmse = np.sqrt(-cross_val_score(ml_pipeline, X, y, cv=kf, scoring='neg_mean_squared_error'))
    
    print(f"  CV R² Score (Accuratezza Modello): {cv_r2.mean():.4f} +/- {cv_r2.std():.4f}")
    print(f"  CV RMSE (Errore di Previsione):     {cv_rmse.mean():.4f} €")
    
    return ml_pipeline


def plot_results(S, payoffs, K):
    print("\nGenerazione della dashboard grafica delle opzioni asiatiche...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=150)
    time_grid = [1/3, 2/3, 1.0]

    # Pannello 1: Percorsi Monte Carlo
    for i in range(50):
        axes[0].plot(time_grid, S[i, :], color='royalblue', alpha=0.3, lw=1.2)
    axes[0].plot(time_grid, np.mean(S[:50, :], axis=0), color='darkred', lw=2.5, label='Media Campionaria')
    axes[0].axhline(K, color='black', linestyle='--', label=f'Strike K = {K}')
    axes[0].set_title('1. Simulazione Monte Carlo: Percorsi Sottostante ($S_t$)', fontsize=11, fontweight='bold')
    axes[0].set_xlabel('Tempo (Anni)')
    axes[0].set_ylabel('Prezzo Azione (€)')
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)

    # Pannello 2: Distribuzione dei Payoff
    axes[1].hist(payoffs['Call_Avg_Price'], bins=50, color='forestgreen', alpha=0.7, label='Call Avg Price Payoff')
    axes[1].hist(payoffs['Put_Avg_Price'], bins=50, color='darkorange', alpha=0.7, label='Put Avg Price Payoff')
    axes[1].set_title('2. Distribuzione dei Payoff delle Opzioni Asiatiche', fontsize=11, fontweight='bold')
    axes[1].set_xlabel('Valore del Payoff a Scadenza (€)')
    axes[1].set_ylabel('Frequenza')
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('opzioni_asiatiche_montecarlo_ml.png', dpi=150)
    plt.show()


if __name__ == '__main__':
    df_paths, S, A, St, payoffs = simulate_asian_options()
    pipeline = run_ml_surrogate_model(df_paths)
    plot_results(S, payoffs, K=100.0)