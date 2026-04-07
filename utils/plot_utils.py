import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def plot_feature_distributions(df, feature_cols, output_path):
    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    for ax, col in zip(axes.flatten(), feature_cols):
        df[col].hist(bins=60, ax=ax, edgecolor='none', alpha=0.75)
        ax.set_title(col)
        ax.set_ylabel('Count')
    plt.suptitle('Feature distributions', y=1.01)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.show()

def plot_correlation_matrix(df, feature_cols, output_path):
    plt.figure(figsize=(10, 8))
    corr = df[feature_cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                square=True, linewidths=0.5)
    plt.title('Feature correlation matrix')
    plt.savefig(output_path, dpi=150)
    plt.show()

def plot_training_curves(history, output_path, title='Model'):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(history.history['loss'],     label='Train loss')
    ax1.plot(history.history['val_loss'], label='Val loss', linestyle='--')
    ax1.set_title(f'{title} — Loss curves')
    ax1.set_xlabel('Epoch')
    ax1.legend()

    ax2.plot(history.history['masked_mape'],     label='Train MAPE')
    ax2.plot(history.history['val_masked_mape'], label='Val MAPE', linestyle='--')
    ax2.axhline(y=8.33, color='red', linestyle=':', label='2022 baseline (8.33%)')
    ax2.set_title(f'{title} — MAPE curves')
    ax2.set_xlabel('Epoch')
    ax2.legend()

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150)
    plt.show()

def plot_predictions(y_true, y_pred_lstm, y_pred_tf, output_path, n_plot=500):
    fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True)

    axes[0].plot(y_true[-n_plot:], label='Actual rainfall', color='steelblue', linewidth=1)
    axes[0].plot(y_pred_lstm[-n_plot:], label='LSTM v2', color='orange', linewidth=1, alpha=0.8)
    axes[0].plot(y_pred_tf[-n_plot:],   label='Transformer', color='crimson', linewidth=1, alpha=0.8)
    axes[0].set_ylabel('Rainfall (mm/hr)')
    axes[0].set_title('Model predictions vs actual rainfall (test set)')
    axes[0].legend()
    axes[0].set_ylim(bottom=0)

    # Residuals
    axes[1].plot(y_true[-n_plot:] - y_pred_lstm[-n_plot:],
                 label='LSTM v2 residual', color='orange', linewidth=0.7, alpha=0.7)
    axes[1].plot(y_true[-n_plot:] - y_pred_tf[-n_plot:],
                 label='Transformer residual', color='crimson', linewidth=0.7, alpha=0.7)
    axes[1].axhline(0, color='black', linewidth=0.5, linestyle='--')
    axes[1].set_ylabel('Residual (mm/hr)')
    axes[1].set_xlabel('Hour index (test set)')
    axes[1].legend()

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150)
    plt.show()

def plot_mape_comparison(results_df, output_path):
    models = results_df['Model']
    mapes  = [float(x) if x != '-' else 0 for x in results_df['MAPE (%)']]
    colors = ['#888', '#E8A020', '#3B8BD4', '#A040C0']
    
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(models, mapes, color=colors[:len(models)], edgecolor='none', alpha=0.85)
    ax.axhline(y=8.33, color='grey', linestyle='--', linewidth=1, label='2022 LSTM (8.33%)')
    ax.set_ylabel('MAPE (%)')
    ax.set_title('Model comparison — Masked MAPE on Ireland test set')
    ax.legend()
    for bar, val in zip(bars, mapes):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                f'{val:.2f}%', ha='center', va='bottom', fontsize=11)
    plt.xticks(rotation=20, ha='right')
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150)
    plt.show()
