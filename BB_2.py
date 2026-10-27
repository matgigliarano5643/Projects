import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, StratifiedKFold, LeaveOneOut, cross_val_score, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, classification_report

def load_and_clean_spectrophotometer_data(filepath: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Esegue l'ETL dal formato grezzo a 63 colonne a un DataFrame tidy (Long Format),
    corregge la deriva di dark-current e applica le equazioni ottiche del prisma.
    """
    xls = pd.ExcelFile(filepath)
    df_raw = pd.read_excel(filepath, sheet_name=xls.sheet_names[0])
    
    voltages = [12, 11, 10, 9, 8, 7, 6]
    # Angoli di zero (picco del raggio non disperso) per ciascuna run sperimentale
    zero_offsets = {12: 12.516, 11: 11.190, 10: 11.575, 9: 12.202, 8: 11.281, 7: 11.278, 6: 10.917}
    
    # Parametri fisici del prisma in vetro flint (PASCO)
    CAUCHY_A = 1.689
    CAUCHY_B = 13900.0  # nm^2
    
    records = []
    summary_stats = []
    
    for v in voltages:
        # Gestione programmatica delle inconsistenze nei nomi delle colonne Excel
        t_col = f'Time (s) {v}V' if v != 7 else 'Time (s) Run 7V'
        a_col = f'Angle (rad) {v}V'
        p_col = f'Position (m) {v}V'
        i_col = f'Relative Intensity (%)  {v}V' if v == 12 else f'Relative Intensity (%) {v}V'
        l_col = f' Lambda {v}V' if v == 12 else f'Lambda {v}V'
        
        sub = df_raw[[t_col, a_col, p_col, i_col, l_col]].dropna().copy()
        sub.columns = ['time_s', 'angle_raw_deg', 'position_m', 'intensity_raw', 'lambda_raw_excel']
        sub['voltage'] = v
        
        # 1. Sottrazione della corrente di buio (Dark-Current Baseline) sui primi 10 campioni (0.5 s)
        baseline = sub['intensity_raw'].iloc[:10].mean()
        sub['intensity_net'] = sub['intensity_raw'] - baseline
        
        # 2. Calibrazione angolare esatta con pi greco in doppia precisione IEEE 754
        sub['theta_deg'] = -sub['angle_raw_deg'] - zero_offsets[v]
        sub['theta_rad'] = np.deg2rad(sub['theta_deg'])
        
        # 3. Derivata numerica temporale: velocità angolare di scansione del braccio (deg/s)
        dt = sub['time_s'].diff().replace(0, np.nan).fillna(0.05)
        sub['angular_velocity'] = sub['theta_deg'].diff().fillna(0.0) / dt
        
        # 4. Calcolo vettorizzato dell'indice di rifrazione n(theta) tramite legge di Snell sul prisma (alpha = 60°)
        sin_term = (2.0 / np.sqrt(3.0)) * np.sin(sub['theta_rad']) + 0.5
        sub['n_prism'] = np.sqrt(sin_term**2 + 0.75)
        
        # 5. Inversione di Cauchy con mascheramento della singolarità iperbolica (n <= 1.689)
        delta_n = sub['n_prism'] - CAUCHY_A
        sub['lambda_sq_excel'] = CAUCHY_B / delta_n
        sub['lambda_nm'] = np.where(delta_n > 0, np.sqrt(CAUCHY_B / delta_n), np.nan)
        
        # 6. Etichettatura fisica dei regimi ottici
        conditions = [
            (sub['theta_deg'].abs() <= 4.0) & (sub['intensity_net'] > 0.5),
            (sub['theta_deg'] > 4.0) & (sub['theta_deg'] < 55.2),
            (sub['theta_deg'] >= 55.2)
        ]
        sub['optical_regime'] = np.select(conditions, [1, 2, 3], default=0)
        
        records.append(sub)
        summary_stats.append({
            'Voltage_V': v,
            'Valid_Samples': len(sub),
            'Dark_Baseline_%': baseline,
            'Max_Direct_Beam_%': sub['intensity_net'].max()
        })
        
    df_clean = pd.concat(records, ignore_index=True)
    df_summary = pd.DataFrame(summary_stats)
    return df_clean, df_summary


def run_ml_pipeline(df_clean: pd.DataFrame):
    voltages = sorted(df_clean['voltage'].unique(), reverse=True)
    theta_grid = np.linspace(55.5, 68.0, 200)
    
    spectral_matrix = []
    gpr_results = []
    
    # --- A. GAUSSIAN PROCESS REGRESSION (Denoising & Incertezza Spettrale) ---
    for v in voltages:
        disp = df_clean[(df_clean['voltage'] == v) & 
                        (df_clean['theta_deg'] >= 55.0) & 
                        (df_clean['theta_deg'] <= 70.0)].sort_values('theta_deg').drop_duplicates('theta_deg')
        
        X_t = disp[['theta_deg']].values
        y_i = disp['intensity_net'].values
        
        kernel = C(1.0, (1e-2, 1e3)) * RBF(length_scale=1.5, length_scale_bounds=(0.3, 10.0)) + \
                 WhiteKernel(noise_level=0.01, noise_level_bounds=(1e-5, 1.0))
        
        gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=5, normalize_y=True, random_state=42)
        gpr.fit(X_t, y_i)
        
        y_pred, y_std = gpr.predict(theta_grid.reshape(-1, 1), return_std=True)
        spectral_matrix.append(y_pred)
        
        idx_max = np.argmax(y_pred)
        theta_pk = theta_grid[idx_max]
        n_pk = np.sqrt(((2.0 / np.sqrt(3.0)) * np.sin(np.deg2rad(theta_pk)) + 0.5)**2 + 0.75)
        lam_pk = np.sqrt(13900.0 / (n_pk - 1.689))
        auc_energy = np.trapezoid(np.maximum(y_pred, 0), theta_grid)
        
        gpr_results.append({
            'Voltage_V': v,
            'Peak_Intensity_Net_%': y_pred[idx_max],
            'Theta_Peak_deg': theta_pk,
            'Lambda_Peak_nm': lam_pk,
            'Spectral_Energy_AUC': auc_energy,
            'Estimated_Sensor_Noise_%': np.sqrt(gpr.kernel_.k2.noise_level),
            'GPR_R2': r2_score(y_i, gpr.predict(X_t))
        })
        
    X_spectra = np.array(spectral_matrix)
    df_gpr = pd.DataFrame(gpr_results)
    
    # --- B. ANALISI DELLE COMPONENTI PRINCIPALI (PCA FUNZIONALE) ---
    pca_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=3, random_state=42))
    ])
    Z_pca = pca_pipe.fit_transform(X_spectra)
    exp_var = pca_pipe.named_steps['pca'].explained_variance_ratio_
    df_gpr['PC1_Score'] = Z_pca[:, 0]
    df_gpr['PC2_Score'] = Z_pca[:, 1]
    
    # --- C. CLASSIFICAZIONE DEI REGIMI OTTICI (RANDOM FOREST) ---
    features_cols = ['voltage', 'theta_deg', 'n_prism', 'angular_velocity']
    X_all = df_clean[features_cols].values
    y_regime = df_clean['optical_regime'].values
    y_intensity = df_clean['intensity_net'].values
    
    rf_clf = RandomForestClassifier(n_estimators=150, max_depth=8, random_state=42)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    clf_acc = cross_val_score(rf_clf, X_all, y_regime, cv=skf, scoring='accuracy')
    
    # --- D. REGRESSIONE GLOBALE & PROBLEMA INVERSO DI CALIBRAZIONE ---
    gbr = GradientBoostingRegressor(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    reg_r2 = cross_val_score(gbr, X_all, y_intensity, cv=kf, scoring='r2')
    reg_rmse = np.sqrt(-cross_val_score(gbr, X_all, y_intensity, cv=kf, scoring='neg_mean_squared_error'))
    
    # Problema Inverso: Stima della Tensione V dallo Spettro (Leave-One-Out CV)
    X_inv = df_gpr[['Peak_Intensity_Net_%', 'Theta_Peak_deg', 'Spectral_Energy_AUC', 'PC1_Score', 'PC2_Score']].values
    y_inv = df_gpr['Voltage_V'].values
    
    inv_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('ridge', Ridge(alpha=0.5))
    ])
    y_inv_pred = cross_val_predict(inv_pipe, X_inv, y_inv, cv=LeaveOneOut())
    df_gpr['Predicted_Voltage_LOOCV'] = y_inv_pred
    
    print("=== RISULTATI GAUSSIAN PROCESS, PCA E CALIBRAZIONE INVERSA ===")
    print(df_gpr.round(4).to_string(index=False))
    print(f"\n[PCA] Varianza Spiegata: PC1={exp_var[0]*100:.2f}%, PC2={exp_var[1]*100:.2f}%, Totale 3 PC={np.sum(exp_var)*100:.2f}%")
    print(f"[Random Forest Regime Classifier] 5-Fold CV Accuracy: {clf_acc.mean()*100:.2f}% +/- {clf_acc.std()*100:.2f}%")
    print(f"[Gradient Boosting Surface Regressor] 5-Fold CV R2: {reg_r2.mean():.4f} | RMSE: {reg_rmse.mean():.4f}%")
    print(f"[Inverse Voltage Calibration LOOCV] R2: {r2_score(y_inv, y_inv_pred):.4f} | MAE: {mean_absolute_error(y_inv, y_inv_pred):.3f} V")

    print("Generazione e apertura della dashboard grafica a 4 pannelli...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=150)
    cmap = plt.cm.plasma(np.linspace(0.85, 0.1, len(voltages)))

    # Panel 1: Curve grezze corrette per la baseline
    ax1 = axes[0, 0]
    for idx, v in enumerate(voltages):
        sub = df_clean[df_clean['voltage'] == v].sort_values('theta_deg')
        base_val = df_summary.loc[df_summary['Voltage_V'] == v, 'Dark_Baseline_%'].values[0]
        ax1.plot(sub['theta_deg'], sub['intensity_net'], label=f'{v}V (Base: {base_val:+.2f}%)', color=cmap[idx], lw=1.8)
    ax1.axvline(0.0, color='gray', linestyle='--', alpha=0.6, label='Rif. Raggio Diretto (0°)')
    ax1.axvline(55.14, color='red', linestyle=':', alpha=0.7, label='Polo di Cauchy (n = 1.689)')
    ax1.set_title('1. Calibrazione Angolare e Sottrazione Dark-Current (12V - 6V)', fontsize=11, fontweight='bold')
    ax1.set_xlabel('Angolo di Deviazione Calibrato $\\theta_{corr}$ (gradi)')
    ax1.set_ylabel('Intensità Relativa Netta (%)')
    ax1.legend(fontsize=8, loc='upper left')
    ax1.grid(True, alpha=0.3)

    # Panel 2: Denoising GPR sui picchi spettrali
    ax2 = axes[0, 1]
    for idx, v in enumerate(voltages):
        sub = df_clean[(df_clean['voltage'] == v) & (df_clean['theta_deg'] >= 55.5) & (df_clean['theta_deg'] <= 66.0)]
        ax2.scatter(sub['theta_deg'], sub['intensity_net'], color=cmap[idx], s=14, alpha=0.5)
        ax2.plot(theta_grid, X_spectra[idx], color=cmap[idx], lw=2.0, label=f'{v}V (Picco GPR: {df_gpr.loc[idx, "Peak_Intensity_Net_%"]:.2f}%)')
        ax2.plot(df_gpr.loc[idx, 'Theta_Peak_deg'], df_gpr.loc[idx, 'Peak_Intensity_Net_%'], 'ko', markersize=5)
    ax2.set_xlim(55.5, 65.0)
    ax2.set_title('2. Regole di Dispersione e Denoising con Gaussian Process (GPR)', fontsize=11, fontweight='bold')
    ax2.set_xlabel('Angolo nel Regime Dispersivo $\\theta_{corr}$ (gradi)')
    ax2.set_ylabel('Intensità Netta Filtrata (%)')
    ax2.legend(fontsize=8, loc='upper right')
    ax2.grid(True, alpha=0.3)

    # Panel 3: Spazio delle Componenti Principali (PCA)
    ax3 = axes[1, 0]
    ax3.scatter(Z_pca[:, 0], Z_pca[:, 1], c=voltages, cmap='plasma', s=120, edgecolors='k', zorder=3)
    for idx, v in enumerate(voltages):
        ax3.annotate(f'{v}V', (Z_pca[idx, 0]+0.25, Z_pca[idx, 1]+0.15), fontsize=9, fontweight='bold')
    ax3.plot(Z_pca[:, 0], Z_pca[:, 1], 'k--', alpha=0.4, zorder=2)
    ax3.set_title(f'3. Riduzione Dimensionale Spettrale (PCA: PC1+PC2 = {np.sum(exp_var[:2])*100:.1f}% var.)', fontsize=11, fontweight='bold')
    ax3.set_xlabel(f'Componente Principale 1 ({exp_var[0]*100:.1f}% varianza)')
    ax3.set_ylabel(f'Componente Principale 2 ({exp_var[1]*100:.1f}% varianza)')
    ax3.grid(True, alpha=0.3)

    # Panel 4: Modello Inverso (Predizione Tensione)
    ax4 = axes[1, 1]
    ax4.scatter(y_inv, y_inv_pred, c=voltages, cmap='plasma', s=100, edgecolors='k', zorder=3, label='Predizione LOOCV (Ridge)')
    ax4.plot([5.5, 12.5], [5.5, 12.5], 'r--', lw=1.5, label=f'Ideale (R² = {r2_score(y_inv, y_inv_pred):.3f}, MAE = {mean_absolute_error(y_inv, y_inv_pred):.2f}V)')
    for v_t, v_p in zip(y_inv, y_inv_pred):
        ax4.annotate(f'{v_t}V ({v_p:.1f}V)', (v_t, v_p), xytext=(v_t-0.4, v_p+0.35), fontsize=8)
    ax4.set_title('4. Modello Inverso di Calibrazione Tensione da Feature Spettrali', fontsize=11, fontweight='bold')
    ax4.set_xlabel('Tensione Reale di Alimentazione $V_{true}$ (Volt)')
    ax4.set_ylabel('Tensione Predetta in LOOCV $V_{pred}$ (Volt)')
    ax4.legend(fontsize=9, loc='upper left')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('spettrofotometro_ml_pipeline.png', dpi=150)
    plt.show()
    
    return df_gpr, X_spectra, theta_grid

if __name__ == '__main__':
    FILE_PATH = 'Esperimento Spettrofotometro 12-6V Grafico corretto.xlsx'
    df_clean, df_summary = load_and_clean_spectrophotometer_data(FILE_PATH)
    df_gpr, X_spectra, theta_grid = run_ml_pipeline(df_clean)