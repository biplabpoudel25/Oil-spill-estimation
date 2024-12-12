import time
import torch
import argparse
import numpy as np
from utils import *
from tqdm import tqdm
from model import SimpleRegressionModel, HuberLoss
from sklearn.metrics import r2_score, mean_squared_error
from torch.utils.data import TensorDataset, DataLoader, random_split


def parse_option():
    parser = argparse.ArgumentParser('Perform training and save the trained checkpoints', add_help=False)
    parser.add_argument('--features-path', type=str, required=True, help='path to the deep features of images')
    parser.add_argument('--model-name', type=str, required=True, default='mobilenetv3', choices=['resnet18', 'mobilenetv3'],
                        help='Name of the model to extract features')
    parser.add_argument('--batch-size', type=int, required=True, default=64, help='batch size of images')
    parser.add_argument('--num-epochs', type=int, required=True, default=1000, help='Num of epochs to train model')
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

        # Calculate the sizes for training and validation splits
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size

        # Split the dataset into training and validation sets
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        # Create DataLoaders for training and validation
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

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

                labels_expanded = torch.zeros((labels.shape[0], 3), device=device)
                mean = labels.to(device)

                margin = 0.15 * torch.abs(mean)
                labels_expanded[:, 0] = mean - margin  # lower bound
                labels_expanded[:, 1] = mean  # mean
                labels_expanded[:, 2] = mean + margin  # upper bound

                outputs = model(images)

                loss = criterion(outputs, labels_expanded)

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

                    labels_expanded = torch.zeros((labels.shape[0], 3), device=device)
                    mean = labels.to(device)
                    margin = 0.15 * torch.abs(mean)
                    labels_expanded[:, 0] = mean - margin
                    labels_expanded[:, 1] = mean
                    labels_expanded[:, 2] = mean + margin

                    outputs = model(images)
                    loss = criterion(outputs, labels_expanded)
                    val_loss += loss.item()

                    # Store labels
                    all_labels.extend(labels.cpu().numpy().tolist())

                    # Separate and store predictions
                    predictions = outputs.cpu().numpy()

                    all_means.extend(predictions[:, 1])

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

        time_end = time.time()
        time_taken = time_end - time_start
        total_time = compute_time(time_taken)
        logger.info(f"\nTotal time taken to train the model: {total_time}")

    except Exception as e:
        logger.info('\n')
        logger.error("An error occurred: %s", str(e), exc_info=True)
        sys.exit(1)