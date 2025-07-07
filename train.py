import argparse
import time
from tqdm import tqdm
import torch
from utils import *
from torch.utils.data import TensorDataset, DataLoader, random_split
import matplotlib.font_manager as fm
from model import *

font_path = os.path.expanduser(r"./Times New Roman Bold.ttf")
times_new_roman = fm.FontProperties(fname=font_path)


def parse_option():
    parser = argparse.ArgumentParser('Perform training and save the trained checkpoints', add_help=False)

    parser.add_argument('--features-path', type=str, required=True, help='path to the deep features of images')
    parser.add_argument('--model-name', type=str, required=True, default='mobilenetv3', choices=['resnet18', 'mobilenetv3'],
                        help='Name of the model to extract features')
    parser.add_argument('--batch-size', type=int, required=True, default=64, help='batch size of images')
    parser.add_argument('--num-epochs', type=int, required=True, default=4000, help='Num of epochs to train model')
    parser.add_argument('--log-dir', type=str, required=True, help='path to save the training log')
    parser.add_argument('--ckpt-name', type=str, required=True, help='name to save model checkpoint')


    args, unparsed = parser.parse_known_args()
    return args



if __name__ == '__main__':
    args = parse_option()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_name = args.model_name

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

        logger.info(f'\nThe shape of a feature is: {features[0].shape}')
        print(model_name)

        all_features = []
        for feature in features:
            if model_name in ['resnet18', 'mobilenetv3']:
                feature = feature.view(-1)

            all_features.append(feature)

        all_features = torch.stack(all_features)
        labels = torch.stack(labels)

        all_features = [min_max_normalize(f) for f in all_features]
        all_features = torch.stack(all_features)

        logger.info(f'Shape of the features is: {all_features.shape}\n')

        # Define the dataset
        dataset = TensorDataset(all_features, labels)

        # Using 70% for training, 15% for validation, 15% for calibration
        train_size = int(0.7 * len(dataset))
        val_size = int(0.15 * len(dataset))
        calib_size = len(dataset) - train_size - val_size

        train_dataset, val_dataset, calib_dataset = random_split(
            dataset,
            [train_size, val_size, calib_size]
        )

        # Create DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        calib_loader = DataLoader(calib_dataset, batch_size=args.batch_size, shuffle=False)

        C = all_features[0].shape[0]

        model = SimpleRegressionModel(C)
        model.to(device)

        params = count_trainable_parameters(model)
        print(f'Number of trainable parameters: {params}')

        criterion = HuberLoss(delta=5.0)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=5e-3)

        train_losses = []
        val_losses = []

        num_epochs = args.num_epochs
        best_val_rmse = float('inf')
        time_start = time.time()

        for epoch in tqdm(range(args.num_epochs)):
            model.train()
            train_loss = 0.0  # initialize training loss

            for i, (images, labels) in enumerate(train_loader):
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                outputs = outputs.squeeze()

                loss = criterion(outputs, labels)

                # Backward and optimize
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()  # Accumulate training loss

            # Store average train loss for the epoch
            avg_train_loss = train_loss / len(train_loader)
            train_losses.append(avg_train_loss)

            # validation phase
            model.eval()
            val_loss = 0.0
            all_labels = []
            all_means = []

            with torch.no_grad():
                for i, (images, labels) in enumerate(val_loader):
                    images, labels = images.to(device), labels.to(device)

                    outputs = model(images)
                    outputs = outputs.squeeze()
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()

                    if torch.is_tensor(outputs):
                        all_labels.extend(labels.cpu().numpy().tolist())

                        if outputs.dim() > 0:
                            all_means.extend(outputs.cpu().numpy().tolist())
                        else:
                            all_means.append(outputs.item())
                    else:
                        all_labels.append(labels.cpu().numpy().item())
                        all_means.append(outputs)

            # Store average val loss for the epoch
            avg_val_loss = val_loss / len(val_loader)
            val_losses.append(avg_val_loss)

            # Convert to numpy arrays
            all_labels = np.array(all_labels)
            all_means = np.array(all_means)

            # Replace negative values with 0
            all_means = np.maximum(0, all_means)

            # Calculate metrics for mean predictions (primary metric)
            val_r2_mean = r2_score(all_labels, all_means)
            val_rmse_mean = np.sqrt(mean_squared_error(all_labels, all_means))

            logger.info(
                f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss / len(train_loader)}, Val Loss: {val_loss / len(val_loader)}, Val R2: {val_r2_mean}, Val RMSE: {val_rmse_mean}")

            # Save the best model checkpoint
            if val_rmse_mean < best_val_rmse:

                best_val_rmse = val_rmse_mean
                ckpt_path = save_checkpoint(model, args.ckpt_name, model_name)
                logger.info(f'Best checkpoint saved at epoch: {epoch + 1} at {ckpt_path}.')

        # After training, fit conformal predictor on calibration set
        logger.info("\nFitting conformal predictor on calibration set...")
        bins, quantiles = fit_local_conformal_predictor(
            model,
            calib_loader,
            n_bins=10,
            alpha=0.05,
            device=device
        )

        # Save conformal prediction parameters
        conformal_params = {
            'bins': bins,
            'quantiles': quantiles
        }

        conformal_path = os.path.join(
            os.path.dirname(ckpt_path),
            f'conformal_params_{model_name}.pt'
        )
        torch.save(conformal_params, conformal_path)
        logger.info(f'Conformal prediction parameters saved at: {conformal_path}')

        # Evaluate conformal predictions on calibration set
        logger.info("\nEvaluating conformal predictions on calibration set...")
        calib_results = evaluate_with_local_conformal(
            model,
            calib_loader,
            bins,
            quantiles,
            device=device
        )

        # Log the evaluation metrics
        logger.info("\nConformal Prediction Performance on Calibration Set:")
        logger.info(f"R2 Score: {calib_results['r2']:.4f}")
        logger.info(f"RMSE: {calib_results['rmse']:.4f}")
        logger.info(f"95% CI Coverage: {calib_results['coverage']:.2f}%")
        logger.info(f"Average CI Width: {calib_results['ci_width']:.4f}")

        # Log metrics by concentration range
        logger.info("\nMetrics by Concentration Range:")
        logger.info("\n" + str(calib_results['bin_metrics']))

        # Log total training time
        time_end = time.time()
        time_taken = time_end - time_start
        total_time = compute_time(time_taken)
        logger.info(f"\nTotal time taken to train the model: {total_time}")

    except Exception as e:
        logger.info('\n')
        logger.error("An error occurred: %s", str(e), exc_info=True)
        sys.exit(1)