import os
import pickle
import sys
import numpy as np
import torch
import logging
import pandas as pd
from PIL import Image
from datetime import datetime
from argparse import Namespace
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.interpolate import make_interp_spline


class Image_Dataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, transform=None):
        self.data_info = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, idx):
        img_path = self.data_info.iloc[idx, 1]
        image = Image.open(img_path).convert('RGB')

        labels = self.data_info.iloc[idx, 0]

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(labels).float()


def log_arguments(args: Namespace, logger: logging.Logger):
    logger.info("\nArguments:")
    for arg, value in vars(args).items():
        logger.info(f"{arg}: {value}")
    logger.info('\n')


def setup_logger(log_filename, model_name):
    # Create logs directory if it doesn't exist
    log_dir = 'logs'
    try:
        os.makedirs(log_dir, exist_ok=True)
    except Exception as e:
        print(f"Error creating log directory: {e}")
        sys.exit(1)

    # Add date and time to the log file name
    base_name, ext = os.path.splitext(log_filename)
    if ext == '':
        ext = '.log'

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    full_log_filename = f"{base_name}_{model_name}_{timestamp}.log"
    log_path = os.path.join(log_dir, full_log_filename)

    # Set up logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # File handler
    try:
        file_handler = logging.FileHandler(log_path)
        file_formatter = logging.Formatter('%(message)s')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    except Exception as e:
        print(f"Error setting up file handler: {e}")
        sys.exit(1)

    # Console handler
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    return logger


def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def min_max_normalize(features):
    min_val = torch.min(features)
    max_val = torch.max(features)
    return (features - min_val) / (max_val - min_val)


def save_checkpoint(model, checkpoint_name, model_name):
    # Create the checkpoints directory if it doesn't exist
    checkpoint_dir = 'checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Create the full path for the checkpoint file
    checkpoint_path = os.path.join(checkpoint_dir, f"{model_name}_{checkpoint_name}.pth")

    # Save only the state dictionary
    torch.save(model.state_dict(), checkpoint_path)

    return checkpoint_path


def compute_time(seconds):
    days, remainder = divmod(seconds, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, seconds = divmod(remainder, 60)

    if days > 0:
        return f"{days:.0f} days, {hours:.0f} hours, {minutes:.0f} minutes, {seconds:.2f} seconds"
    elif hours > 0:
        return f"{hours:.0f} hours, {minutes:.0f} minutes, {seconds:.2f} seconds"
    elif minutes > 0:
        return f"{minutes:.0f} minutes, {seconds:.2f} seconds"
    else:
        return f"{seconds:.2f} seconds"
    

def fit_local_conformal_predictor(model, calibration_loader, n_bins=10, alpha=0.05, device='cuda'):
    """
    Fit conformal predictor with locally adaptive intervals
    """
    model.eval()
    predictions = []
    errors = []

    with torch.no_grad():
        for images, labels in calibration_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            preds = outputs.squeeze()

            predictions.extend(preds.cpu().numpy())
            errors.extend(torch.abs(preds - labels).cpu().numpy())

    predictions = np.array(predictions)
    errors = np.array(errors)

    # Create bins based on prediction magnitude
    bins = np.linspace(0, max(predictions), n_bins + 1)
    bin_indices = np.digitize(predictions, bins) - 1

    # Calculate quantiles for each bin
    quantiles = []
    for i in range(n_bins):
        bin_errors = errors[bin_indices == i]
        if len(bin_errors) > 0:
            quantiles.append(np.quantile(bin_errors, 1 - alpha))
        else:
            quantiles.append(np.quantile(errors, 1 - alpha))  # fallback

    return bins, np.array(quantiles)


def predict_with_local_conformal(model, images, bins, quantiles, device='cuda', max_concentration=500):
    """
    Make predictions with locally adaptive intervals
    """
    model.eval()
    with torch.no_grad():
        outputs = model(images)

        # Ensure predictions are = 0
        clamped_preds = torch.clamp(outputs.squeeze(), min=0)
        mean_pred = clamped_preds.cpu().numpy()

        #        mean_pred = outputs.squeeze().cpu().numpy()

        # Find appropriate bin for each prediction
        bin_indices = np.digitize(mean_pred, bins) - 1
        bin_indices = np.clip(bin_indices, 0, len(quantiles) - 1)

        # Get corresponding quantiles
        prediction_quantiles = quantiles[bin_indices]

        lower_bound = np.maximum(mean_pred - prediction_quantiles, 0)
        upper_bound = mean_pred + prediction_quantiles

        return mean_pred, lower_bound, upper_bound

def evaluate_with_local_conformal(model, test_loader, bins, quantiles, device='cuda', max_concentration=500):
    """
    Evaluate model with local conformal prediction intervals
    """
    model.eval()

    all_labels = []
    all_means = []
    all_lower_bounds = []
    all_upper_bounds = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            # Get predictions with local conformal intervals
            mean_pred, lower_bound, upper_bound = predict_with_local_conformal(
                model, images, bins, quantiles, device, max_concentration
            )

            # Store results
            all_labels.extend(labels.cpu().numpy().tolist())
            all_means.extend(mean_pred.tolist())
            all_lower_bounds.extend(lower_bound.tolist())
            all_upper_bounds.extend(upper_bound.tolist())

    # Convert to numpy arrays
    all_labels = np.array(all_labels)
    all_means = np.array(all_means)
    all_lower_bounds = np.array(all_lower_bounds)
    all_upper_bounds = np.array(all_upper_bounds)

    # Create DataFrame with results
    results_df = pd.DataFrame({
        'True_Concentration': all_labels,
        'Predicted_Concentration': all_means,
        'Lower_Bound': all_lower_bounds,
        'Upper_Bound': all_upper_bounds
    })

    # Calculate interval width
    results_df['Interval_Width'] = results_df['Upper_Bound'] - results_df['Lower_Bound']

    # Calculate if true value falls within the confidence interval
    results_df['Within_CI'] = (results_df['True_Concentration'] >= results_df['Lower_Bound']) & \
                              (results_df['True_Concentration'] <= results_df['Upper_Bound'])

    # Calculate metrics by concentration range
    results_df['Concentration_Bin'] = pd.cut(results_df['True_Concentration'],
                                             bins=5,
                                             labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])

    bin_metrics = results_df.groupby('Concentration_Bin').agg({
        'Within_CI': 'mean',
        'Interval_Width': 'mean'
    })
    bin_metrics['Within_CI'] = bin_metrics['Within_CI'] * 100

    # Calculate overall metrics
    r2 = r2_score(all_labels, all_means)
    rmse = np.sqrt(mean_squared_error(all_labels, all_means))
    coverage = results_df['Within_CI'].mean() * 100
    ci_width = results_df['Interval_Width'].mean()

    # Print results
    print(f"\nEvaluation Results:")
    print(f"R2 Score: {r2:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"95% CI Coverage: {coverage:.2f}%")
    print(f"Average CI Width: {ci_width:.4f}")

    print("\nMetrics by Concentration Range:")
    print(bin_metrics)

    # Print first few rows of the DataFrame
    print("\nPrediction Details:")
    pd.set_option('display.float_format', '{:.2f}'.format)
    print(results_df.head(10))

    #    # Save DataFrame to CSV
    #    results_df.to_csv('local_conformal_prediction_results.csv', index=False)

    return {
        'r2': r2,
        'rmse': rmse,
        'coverage': coverage,
        'ci_width': ci_width,
        'predictions': {
            'true': all_labels,
            'mean': all_means,
            'lower': all_lower_bounds,
            'upper': all_upper_bounds
        },
        'dataframe': results_df,
        'bin_metrics': bin_metrics
    }


def plot_conformal_results(predictions, max_concentration=500):
    """
    Plot results with conformal prediction intervals
    """
    plt.figure(figsize=(12, 6))

    # Sort by true values for better visualization
    sort_idx = np.argsort(predictions['true'])
    true_sorted = predictions['true'][sort_idx]
    mean_sorted = predictions['mean'][sort_idx]
    lower_sorted = predictions['lower'][sort_idx]
    upper_sorted = predictions['upper'][sort_idx]

    # Plot predictions with confidence intervals
    plt.plot(true_sorted, mean_sorted, 'b.', label='Predictions', alpha=0.5, markersize=4)
    plt.fill_between(
        true_sorted,
        lower_sorted,
        upper_sorted,
        alpha=0.2,
        color='blue',
        label='95% CI'
    )

    # Plot perfect prediction line
    plt.plot([0, max_concentration], [0, max_concentration], 'r--', label='Perfect Prediction')

    plt.xlabel('True Concentration')
    plt.ylabel('Predicted Concentration')
    plt.title('Predictions with Conformal Prediction Intervals')
    plt.legend()
    plt.grid(False)

#    plt.show()


def plot_smooth_ci(all_labels, all_means, all_lower_bounds, all_upper_bounds, model_name, plots_dir, font=None, indices_file=None):
    """
        Creates CI plot using either randomly selected points or loaded indices

    """

    # Create results directory if it doesn't exist
    results_dir = os.path.join(plots_dir, 'conformal_results')
    os.makedirs(results_dir, exist_ok=True)

    # Get sampling indices either from file or by random selection
    if indices_file is not None and os.path.exists(indices_file):
        print(f"Loading indices from {indices_file}")
        with open(indices_file, 'rb') as f:
            concentration_indices = pickle.load(f)
        sampled_indices = list(concentration_indices.values())
    else:
        print("Selecting random samples for each concentration")
        # Select random samples and create smooth CI
        unique_concentrations = np.unique(all_labels)
        sampled_indices = []
        concentration_indices = {}

        for conc in unique_concentrations:
            conc_indices = np.where(all_labels == conc)[0]
            selected_idx = np.random.choice(conc_indices)
            sampled_indices.append(selected_idx)
            concentration_indices[conc] = selected_idx

        # Save the indices for future use
        indices_path = os.path.join(results_dir, f'sampled_indices_{model_name}.pkl')
        with open(indices_path, 'wb') as f:
            pickle.dump(concentration_indices, f)

    # Get CI bounds from sampled points
    sampled_labels = all_labels[sampled_indices]
    sampled_lower = all_lower_bounds[sampled_indices]
    sampled_upper = all_upper_bounds[sampled_indices]

    # Sort for interpolation
    sort_idx = np.argsort(sampled_labels)
    sorted_labels = sampled_labels[sort_idx]
    sorted_lower = sampled_lower[sort_idx]
    sorted_upper = sampled_upper[sort_idx]

    # Interpolate CI bounds for all points
    lower_interp = np.interp(all_labels, sorted_labels, sorted_lower)
    upper_interp = np.interp(all_labels, sorted_labels, sorted_upper)

    # Calculate interval widths for all points
    interval_widths = upper_interp - lower_interp

    # Calculate overall metrics
    within_bounds = (all_labels >= lower_interp) & (all_labels <= upper_interp)
    coverage = np.mean(within_bounds) * 100
    avg_width = np.mean(interval_widths)

    # Define concentration ranges and their labels
    ranges = [
        (0, 100, "Very Low"),
        (100, 200, "Low"),
        (200, 300, "Medium"),
        (300, 400, "High"),
        (400, 500, "Very High")
    ]

    # Create DataFrame for metrics
    metrics_data = []
    detailed_metrics = {}
    for low, high, label in ranges:
        mask = (all_labels >= low) & (all_labels <= high)
        if np.any(mask):
            # Coverage calculation
            range_within = np.mean(within_bounds[mask]) * 100
            range_width = np.mean(interval_widths[mask])
            n_samples = np.sum(mask)
            n_covered = np.sum(within_bounds[mask])

            metrics_data.append({
                'Concentration_Bin': label,
                'Within_CI': range_within,
                'Interval_Width': range_width
            })

            detailed_metrics[label] = {
                'range': (low, high),
                'coverage': range_within,
                'width': range_width,
                'n_samples': n_samples,
                'n_covered': n_covered
            }

    metrics_df = pd.DataFrame(metrics_data)

    # Create smooth curves for plotting
    x_smooth = np.linspace(sorted_labels.min(), sorted_labels.max(), 300)
    spl_lower = make_interp_spline(sorted_labels, sorted_lower, k=2)
    spl_upper = make_interp_spline(sorted_labels, sorted_upper, k=2)
    lower_smooth = spl_lower(x_smooth)
    upper_smooth = spl_upper(x_smooth)

    # Plot
    plt.figure(figsize=(8, 6))
    plt.fill_between(x_smooth, lower_smooth, upper_smooth,
                     color='blue', alpha=0.2)
    plt.plot(x_smooth, lower_smooth, 'b-', alpha=0.7,
             label='Lower Bound', linewidth=1)
    plt.plot(x_smooth, upper_smooth, 'g-', alpha=0.7,
             label='Upper Bound', linewidth=1)
    plt.plot([0, 500], [0, 500], 'k:', label='True', linewidth=2)
    plt.scatter(all_labels, all_means, color='red', s=5,
                alpha=1.0, label='Predictions')

    plt.xlabel('Ground truth (mg/L)', fontsize=12, fontweight='bold', fontproperties=font)
    plt.ylabel('Predictions (mg/L)', fontsize=12, fontweight='bold', fontproperties=font)

    plt.xticks(fontsize=12, fontweight='bold', fontproperties=font)
    plt.yticks(fontsize=12, fontweight='bold', fontproperties=font)

    plt.xlim(-20, 520)
    plt.ylim(-20, 580)
    plt.legend(prop=font)
    plt.tight_layout()

    plt.savefig(os.path.join(plots_dir, f'05_27_smooth_confidence_interval_{model_name}.png'), dpi=1200,
                bbox_inches='tight')
    plt.close()

    # Print metrics in desired format
    print("\nConformal Prediction Metrics:")
    print(f"Coverage: {coverage:.2f}%")
    print(f"Average CI Width: {avg_width:.2f}")
    print("\nMetrics by Concentration Range:")
    print(metrics_df.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    return {
        'overall_coverage': coverage,
        'overall_width': avg_width,
        'metrics_df': metrics_df
    }