import time
import torch
import argparse
import numpy as np
from utils import *
from torch import nn
from model import SimpleRegressionModel
from collections import defaultdict
from matplotlib import pyplot as plt
from scipy.interpolate import make_interp_spline
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


def parse_option():
    parser = argparse.ArgumentParser('Perform test and save the results', add_help=False)
    parser.add_argument('--features-path', type=str, required=True, help='path to the deep features of images to test')
    parser.add_argument('--batch-size', type=int, required=True, default=1, help='batch size of images')
    parser.add_argument('--trained-ckpt', type=str, required=True, help='path to trained checkpoints')
    parser.add_argument('--log-dir', type=str, required=True, help='path to save the test log and results')

    args, unparsed = parser.parse_known_args()
    return args


if __name__ == '__main__':
    args = parse_option()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_name = args.features_path.split('/')[-2]

    try:
        logger = setup_logger(args.log_dir, model_name)
    except Exception as e:
        print(f"Failed to set up logger: {e}")
        sys.exit(1)

    logger.info("Log started at: %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    # Log the arguments
    log_arguments(args, logger)

    logger.info(f'Device: {device}')

    try:
        feature_dict = torch.load(args.features_path)
        # extract the features and labels from the features dict
        features = feature_dict['features']
        labels = feature_dict['labels']

        logger.info(f'\nThe shape of the features is: {features[0].shape}')

        all_features = []
        for feature in features:
            if model_name in ['resnet18', 'mobilenetv3']:
                feature = feature.view(-1)
            else:
                raise ValueError('Model not defined!!!')

            all_features.append(feature)

        all_features = torch.stack(all_features)
        labels = torch.stack(labels)

        all_features = [min_max_normalize(f) for f in all_features]
        all_features = torch.stack(all_features)

        test_dataset = TensorDataset(all_features, labels)
        data_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

        C = all_features[0].shape[0]
        model = SimpleRegressionModel(C)
        model.load_state_dict(torch.load(args.trained_ckpt))
        model.to(device)
        print(model)

        model.eval()
        all_labels = []
        all_lower_bounds = []
        all_means = []
        all_upper_bounds = []

        # Dictionary to store predictions for each concentration
        concentration_predictions = defaultdict(lambda: {
            'true': [],
            'predicted': [],
            'rmse': [],
            'lower': [],
            'upper': []
        })

        with torch.no_grad():
            for i, (images, labels) in enumerate(data_loader):
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)

                # Separate and store predictions
                predictions = outputs.cpu().numpy()
                labels_np = labels.cpu().numpy()

                # Store each type of prediction
                all_labels.extend(labels_np)
                all_lower_bounds.extend(predictions[:, 0])
                all_means.extend(predictions[:, 1])
                all_upper_bounds.extend(predictions[:, 2])

                # Store predictions for each concentration
                for label, pred in zip(labels_np, predictions):
                    concentration_predictions[label]['true'].append(label)
                    concentration_predictions[label]['predicted'].append(max(0, pred[1]))  # mean
                    concentration_predictions[label]['lower'].append(max(0, pred[0]))  # lower bound
                    concentration_predictions[label]['upper'].append(max(0, pred[2]))  # upper bound
                    concentration_predictions[label]['rmse'].append(
                        np.sqrt(mean_squared_error([label], [max(0, pred[1])]))
                    )

        # Convert to numpy arrays and handle negative values
        all_labels = np.array(all_labels)
        all_lower_bounds = np.maximum(0, np.array(all_lower_bounds))
        all_means = np.maximum(0, np.array(all_means))
        all_upper_bounds = np.maximum(0, np.array(all_upper_bounds))

        test_r2 = r2_score(all_labels, all_means)
        test_rmse = np.sqrt(mean_squared_error(all_labels, all_means))

        # Create DataFrame
        df = pd.DataFrame({
            'True Value': all_labels,
            'Mean': all_means,
            'Interval': [f'[{low:.2f} - {up:.2f}]'
                         for low, up in zip(all_lower_bounds, all_upper_bounds)]
        })

        # Log the DataFrame
        logger.info('\nDataFrame Contents:\n%s', df.to_string(index=False))

        # Calculate average RMSE for each concentration
        concentration_rmse = {
            conc: np.mean(data['rmse'])
            for conc, data in concentration_predictions.items()
        }

        # Sort values for the true line
        sorted_indices = np.argsort(all_labels)
        sorted_true = all_labels[sorted_indices]
        sorted_predicted = all_means[sorted_indices]

        # 1. Plot true vs Predicted scatter plot
        plt.figure(figsize=(8, 6))
        plt.plot(sorted_true, sorted_true, c='blue', alpha=0.7, label='True', linewidth=2)
        plt.scatter(all_labels, all_means, c='red', alpha=0.7, s=10, label='Predicted')
        plt.xlabel('Ground truth (mg/L)')
        plt.ylabel('Predictions (mg/L)')
        plt.legend()
        plt.grid(False)
        plt.tight_layout()
        # plt.savefig(f'true_vs_pred_{model_name}.png', dpi=1200)
        plt.show()

        # 2. RMSE vs Concentration Plot
        concentrations = sorted(concentration_rmse.keys())
        rmse_values = [concentration_rmse[c] for c in concentrations]

        plt.figure(figsize=(8, 6))
        plt.bar(concentrations, rmse_values, width=1.0, color='red')
        plt.axhline(y=test_rmse, color='blue', linestyle='--')
        plt.xlabel('Ground truths (mg/L)')
        plt.ylabel('Average RMSE')
        # plt.legend()
        plt.tight_layout()
        # plt.savefig(f'average_RMSE_{model_name}.png', dpi=1200)
        plt.show()

        # 3. Box Plot
        # Define concentration values
        concentrations = [0, 10, 20, 50, 80, 100, 150, 200, 250, 300, 350, 400, 450, 500]
        predictions_by_concentration = []

        # Group predictions by concentration
        for conc in concentrations:
            mask = all_labels == conc
            if np.any(mask):
                predictions_by_concentration.append(all_means[mask])
            else:
                predictions_by_concentration.append([])

        plt.figure(figsize=(8, 6))
        plt.boxplot(predictions_by_concentration, labels=concentrations, patch_artist=True, widths=0.4)
        plt.xlabel('Ground truth (mg/L)')
        plt.ylabel('Predictions (mg/L)')
        plt.tight_layout()
        # plt.savefig(f'predicted_boxx_plot_{model_name}.png', dpi=1200)
        plt.show()

        # 4. plot the confidence interval
        # Sort all data by true values for continuous lines
        sort_idx = np.argsort(all_labels)
        sorted_labels = all_labels[sort_idx]
        sorted_means = all_means[sort_idx]
        sorted_lower = all_lower_bounds[sort_idx]
        sorted_upper = all_upper_bounds[sort_idx]

        # Remove duplicates by averaging y values for same x values
        unique_labels = np.unique(sorted_labels)
        averaged_means = np.array([np.mean(sorted_means[sorted_labels == x]) for x in unique_labels])
        averaged_lower = np.array([np.mean(sorted_lower[sorted_labels == x]) for x in unique_labels])
        averaged_upper = np.array([np.mean(sorted_upper[sorted_labels == x]) for x in unique_labels])

        # Create smooth spline functions
        x_smooth = np.linspace(sorted_labels.min(), sorted_labels.max(), 300)  # Automatically span the entire range
        spl_lower = make_interp_spline(unique_labels, averaged_lower, k=2)
        spl_upper = make_interp_spline(unique_labels, averaged_upper, k=2)
        spl_means = make_interp_spline(unique_labels, averaged_means, k=2)

        # Generate smooth curves
        lower_smooth = spl_lower(x_smooth)
        upper_smooth = spl_upper(x_smooth)
        mean_smooth = spl_means(x_smooth)

        # Plot confidence bounds with filled region
        plt.figure(figsize=(8, 6))
        plt.fill_between(x_smooth, lower_smooth, upper_smooth, color='blue', alpha=0.2)
        plt.plot(x_smooth, lower_smooth, 'b-', alpha=0.7, label='Lower Bound', linewidth=2)
        plt.plot(x_smooth, upper_smooth, 'g-', alpha=0.7, label='Upper Bound', linewidth=2)

        # Plot mean predictions
        plt.scatter(all_labels, all_means, color='red', s=10,
                    alpha=0.8, label='Predictions')

        # Plot average prediction line
        plt.plot(x_smooth, mean_smooth, 'black', linestyle='--',
                 linewidth=2, label='Average Prediction')

        plt.xlabel('Ground truth (mg/L)')
        plt.ylabel('Predictions (mg/L)')

        # Set axes limits explicitly with padding
        plt.xlim(-20, 520)  # Adjust x-axis padding
        plt.ylim(-20, 580)  # Adjust y-axis padding
        plt.legend()
        plt.tight_layout()
        # plt.savefig(f'confidence_interval_{model_name}.png', dpi=1200)
        plt.show()

        test_mae = mean_absolute_error(all_labels, all_means)
        non_zero_mask = all_labels != 0
        mape = np.mean(
            np.abs((all_labels[non_zero_mask] - all_means[non_zero_mask]) / all_labels[non_zero_mask])) * 100

        logger.info(f"Overall Test RMSE: {test_rmse:.4f}")
        logger.info(f"R2 Score: {test_r2:.4f}")
        logger.info(f"MAE: {test_mae:.4f}")
        logger.info(f"MAPE: {mape:.4f}")

    except Exception as e:
        logger.info('\n')
        logger.error("An error occurred: %s", str(e), exc_info=True)
        sys.exit(1)
