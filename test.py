import argparse
import torch
from utils import *
from torch.utils.data import DataLoader, TensorDataset
import pickle
import matplotlib.font_manager as fm
from model import *

font_path = os.path.expanduser(r"./Times New Roman Bold.ttf")
times_new_roman = fm.FontProperties(fname=font_path)


def parse_option():
    parser = argparse.ArgumentParser('Perform test and save the results', add_help=False)
    parser.add_argument('--features-path', type=str, required=True, help='path to the deep features of images to test')
    parser.add_argument('--batch-size', type=int, required=False, default=64, help='batch size of images')
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

            all_features.append(feature)

        all_features = torch.stack(all_features)
        labels = torch.stack(labels)

        all_features = [min_max_normalize(f) for f in all_features]
        all_features = torch.stack(all_features)

        test_dataset = TensorDataset(all_features, labels)
        data_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

        # Restore the model and optimizer state
        model = SimpleRegressionModel(input_size=47040)
        model.load_state_dict(torch.load(args.trained_ckpt))

        # Load conformal parameters
        conformal_path = os.path.join(
            os.path.dirname(args.trained_ckpt),
            f'conformal_params_{model_name}.pt'
        )

        try:
            conformal_params = torch.load(conformal_path)
            # Convert lists back to numpy arrays if needed
            bins = np.array(conformal_params['bins'])
            quantiles = np.array(conformal_params['quantiles'])
            logger.info(f"Loaded conformal parameters from: {conformal_path}")
        except Exception as e:
            logger.error(f"Error loading conformal parameters: {e}")
            logger.info("Attempting to load with safer method...")
            try:
                # Alternative loading method

                with open(conformal_path, 'rb') as f:
                    conformal_params = pickle.load(f)
                bins = np.array(conformal_params['bins'])
                quantiles = np.array(conformal_params['quantiles'])
            except Exception as e2:
                logger.error(f"Both loading methods failed. Error: {e2}")
                sys.exit(1)

        model.to(device)
        model.eval()

        # Get all predictions and confidence intervals in one pass
        test_results = evaluate_with_local_conformal(
            model, data_loader, bins, quantiles
        )

        # Extract all required metrics from test_results
        all_labels = test_results['predictions']['true']
        all_means = test_results['predictions']['mean']
        all_lower_bounds = test_results['predictions']['lower']
        all_upper_bounds = test_results['predictions']['upper']

        # Calculate overall metrics
        test_r2 = test_results['r2']
        test_rmse = test_results['rmse']
        test_mae = mean_absolute_error(all_labels, all_means)
        non_zero_mask = all_labels != 0
        mape = np.mean(np.abs((all_labels[non_zero_mask] - all_means[non_zero_mask]) / all_labels[non_zero_mask])) * 100

        # Create plots directory
        plots_dir = os.path.join(args.log_dir, 'HUBER_plots')
        os.makedirs(plots_dir, exist_ok=True)

        # 1. True vs Predicted Plot
        plt.figure(figsize=(8, 6))
        plt.plot(all_labels, all_labels, c='blue', alpha=0.7, label='True', linewidth=2)
        plt.scatter(all_labels, all_means, c='red', alpha=0.7, s=10, label='Predicted')
        plt.xlabel('Ground truth (mg/L)', fontsize=12, fontweight='bold', fontproperties=times_new_roman)
        plt.ylabel('Predictions (mg/L)', fontsize=12, fontweight='bold', fontproperties=times_new_roman)

        plt.xticks(fontsize=12, fontweight='bold', fontproperties=times_new_roman)
        plt.yticks(fontsize=12, fontweight='bold', fontproperties=times_new_roman)

        plt.legend(prop=times_new_roman)
        plt.grid(False)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f'true_vs_pred_{model_name}.png'), dpi=1200)
        plt.close()

        # 2. RMSE vs Concentration Plot
        unique_concentrations = np.unique(all_labels)
        concentration_rmse = {}
        for conc in unique_concentrations:
            mask = all_labels == conc
            conc_predictions = all_means[mask]
            conc_true = all_labels[mask]
            concentration_rmse[conc] = np.sqrt(mean_squared_error(conc_true, conc_predictions))

        plt.figure(figsize=(8, 6))
        plt.bar(list(concentration_rmse.keys()), list(concentration_rmse.values()), width=1.0, color='red')
        plt.axhline(y=test_rmse, color='blue', linestyle='--')
        plt.xlabel('Ground truths (mg/L)', fontsize=12, fontweight='bold', fontproperties=times_new_roman)
        plt.ylabel('Average RMSE', fontsize=12, fontweight='bold', fontproperties=times_new_roman)

        plt.xticks(fontsize=12, fontweight='bold', fontproperties=times_new_roman)
        plt.yticks(fontsize=12, fontweight='bold', fontproperties=times_new_roman)

        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f'average_RMSE_{model_name}.png'), dpi=1200)
        plt.close()

        # 3. Box Plot
        concentrations = [0, 10, 20, 50, 80, 100, 150, 200, 250, 300, 350, 400, 450, 500]
        predictions_by_concentration = [all_means[all_labels == conc] for conc in concentrations]

        plt.figure(figsize=(8, 6))
        plt.boxplot(predictions_by_concentration, labels=concentrations, patch_artist=True, widths=0.4)
        plt.xlabel('Ground truth (mg/L)', fontsize=12, fontweight='bold', fontproperties=times_new_roman)
        plt.ylabel('Predictions (mg/L)', fontsize=12, fontweight='bold', fontproperties=times_new_roman)

        plt.xticks(fontsize=12, fontweight='bold', fontproperties=times_new_roman)
        plt.yticks(fontsize=12, fontweight='bold', fontproperties=times_new_roman)

        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f'predicted_box_plot_{model_name}.png'), dpi=1200)
        plt.close()

        indices_file = None
        ci_metrics = plot_smooth_ci(all_labels, all_means, all_lower_bounds, all_upper_bounds,
                                    model_name, plots_dir, font=times_new_roman, indices_file=indices_file)

        # Print metrics
        print(f"Overall Test RMSE: {test_rmse:.4f}")
        print(f"R2 Score: {test_r2}")
        print(f"MAE: {test_mae:.4f}")
        print(f"MAPE: {mape}")

        unique_concentrations = np.sort(np.unique(all_labels))
        selected_concentrations = unique_concentrations[::2]  # Take every other concentration

        sampled_indices = []
        for conc in selected_concentrations:
            conc_indices = np.where(all_labels == conc)[0]
            sampled_indices.append(np.random.choice(conc_indices, 1)[0])

        sampled_indices = np.array(sampled_indices)

        # Get values for sampled points
        true_conc = all_labels[sampled_indices]
        residuals = all_labels[sampled_indices] - all_means[sampled_indices]
        lower_res = all_lower_bounds[sampled_indices] - all_means[sampled_indices]
        upper_res = all_upper_bounds[sampled_indices] - all_means[sampled_indices]

        plt.figure(figsize=(8, 6))

        # Plot confidence intervals
        plt.vlines(x=true_conc, ymin=lower_res, ymax=upper_res,
                   color='gray', alpha=0.4, linewidth=4, label='95% Confidence Interval')

        # Plot arrows and collect handles for legend
        within_ci_arrow = None
        outside_ci_arrow = None

        for i, conc in enumerate(true_conc):
            residual = residuals[i]
            within_ci = (lower_res[i] <= residual) and (residual <= upper_res[i])
            color = 'green' if within_ci else 'red'

            arrow = plt.arrow(x=conc, y=0, dx=0, dy=residual,
                              color=color, head_width=8, head_length=2,
                              length_includes_head=True, zorder=3)

            if within_ci and within_ci_arrow is None:
                within_ci_arrow = arrow
            elif not within_ci and outside_ci_arrow is None:
                outside_ci_arrow = arrow

        plt.axhline(0, color='black', linestyle='--', linewidth=0.8, label='Zero Residual')

        # Create legend
        legend_elements = [
            plt.Line2D([0], [0], color='gray', alpha=0.4, linewidth=4, label='95% Confidence Interval'),
            plt.Line2D([0], [0], color='black', linestyle='--', label='Zero Residual')
        ]

        if within_ci_arrow is not None:
            legend_elements.append(plt.Line2D([0], [0], color='green', label='Within CI'))
        if outside_ci_arrow is not None:
            legend_elements.append(plt.Line2D([0], [0], color='red', label='Outside CI'))

        plt.legend(handles=legend_elements, loc='upper left', prop=times_new_roman)
        plt.xlabel('Ground truth (mg/L)', fontsize=12, fontweight='bold', fontproperties=times_new_roman)
        plt.ylabel('Residual (True - Predicted)', fontsize=12, fontweight='bold', fontproperties=times_new_roman)

        plt.xticks(fontsize=12, fontweight='bold', fontproperties=times_new_roman)
        plt.yticks(fontsize=12, fontweight='bold', fontproperties=times_new_roman)

        plt.grid(False)
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f'HUBER_residual_plot_{model_name}.png'), dpi=1200)
        plt.close()

    except Exception as e:
        logger.info('\n')
        logger.error("An error occurred: %s", str(e), exc_info=True)
        sys.exit(1)