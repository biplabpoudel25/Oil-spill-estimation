import timm
import torch
import argparse
from utils import *
from torch.utils.data import DataLoader
from torchvision import models, transforms


def parse_option():
    parser = argparse.ArgumentParser('Extract the features of the images and save them as .pt file', add_help=False)
    parser.add_argument('--input-dir', type=str, required=True, help='Input path to the images, given as csv file')
    parser.add_argument('--model', type=str, required=False, default='mobilenetv3', choices=['resnet18', 'mobilenetv3'],
                        help='Name of the model to extract features')
    parser.add_argument('--save-name', type=str, required=True, help='path to save the features, will be '
                                                                     'saved as {model}_{save_name}.pt')
    parser.add_argument('--batch_size', type=int, default=1, required=False, help='batch size of the images')

    args, unparsed = parser.parse_known_args()
    return args


if __name__ == '__main__':
    args = parse_option()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device:', device)

    if args.model == 'resnet18':
        model = timm.create_model('resnet18', pretrained=True)
        layer_to_extract = model.layer4  # layers can be any index of the layers
    elif args.model == 'mobilenetv3':
        model = models.mobilenet_v3_large(pretrained=True)
        layer_to_extract = model.features[-1]
    else:
        raise ValueError(f"{args.model} model is not implemented !!")

    model.to(device)
    model.eval()

    print(f'Using {args.model} model to extract features !!!')

    if args.model in ['resnet18', 'mobilenetv3']:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
        ])
    else:
        raise ValueError(f"{args.model} model is not implemented !!")

    dataset = Image_Dataset(csv_file=args.input_dir, transform=transform)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)


    def extract_intermediate_features(data_loader, model, layer_to_extract):

        # Define a hook function to capture the output of a specific layer
        def hook_fn(module, input, output):
            # You can process and save the features here
            features.append(output)

        # Register the hook to the desired layer
        features = []  # To store the extracted features
        hook = layer_to_extract.register_forward_hook(hook_fn)

        # Extract features for all images in the dataset
        data_features = []
        data_labels = []

        for batch_idx, (input_data, label) in enumerate(data_loader):
            with torch.no_grad():
                if args.model in ['resnet18', 'mobilenetv3']:
                    _ = model(input_data.to(device))

                data_features.extend(features)  # Extend features with the current batch
                data_labels.extend(label)
                features = []  # Clear the features list for the next batch

        # Remove the hook to avoid affecting future forward passes
        hook.remove()

        all_features = []
        for batch_features in data_features:
            all_features.extend(batch_features)

        return all_features, data_labels

features, labels = extract_intermediate_features(data_loader, model, layer_to_extract)

# Create a dictionary to store the features and labels
data_dict = {
    'features': features,
    'labels': labels
}

main_dir = f'deep_features/{args.model}'
os.makedirs(main_dir, exist_ok=True)

filename = f'{args.save_name}.pt'
torch.save(data_dict, os.path.join(main_dir, filename))

print(f'Finished saving the features and labels in {filename} !!! ')