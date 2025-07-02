import argparse
import os
import torch
import wandb

from itertools import chain

from tqdm import tqdm, trange

from utils.loader_VisA import get_data_loader
from utils.general_utils import set_seeds

from models.teacher import FeatureExtractor
from models.students import FeatureProjectionMLP

def train(args):

    set_seeds()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_name = f'{args.label}_{args.class_name}_{args.epochs_no}ep_{args.batch_size}bs'

    wandb.init(
        project = 'L2BT',
        name = model_name,
        # mode = "disabled"
    )

    # Dataloader.
    train_loader, _ = get_data_loader(
        "train", class_name = args.class_name, 
        img_size = args.img_size, batch_size = args.batch_size,
        dataset_path = args.dataset_path)    

    # Feature extractor.
    fe = FeatureExtractor(layers = [7,11]).to(device).eval() 

    # Model instantiation. 
    backward_net = FeatureProjectionMLP(in_features = fe.embed_dim, out_features = fe.embed_dim).to(device)
    forward_net = FeatureProjectionMLP(in_features = fe.embed_dim, out_features = fe.embed_dim).to(device)

    optimizer = torch.optim.Adam(params = chain(backward_net.parameters(), forward_net.parameters()), lr = 1e-4)

    cos_sim = torch.nn.CosineSimilarity(dim = -1, eps = 1e-06)

    for _ in trange(args.epochs_no, desc = f'Training students...'):

        # ------------ [Training Loop] ------------ #

        global_loss = []

        for rgb in tqdm(train_loader, desc = f'    Extracting features from class: {args.class_name}.'):     
            backward_net.train(), forward_net.train()
        
            rgb = rgb.to(device)

            # 1. Feature extraction.
            with torch.no_grad():
                earlier_patch, later_patch = fe(rgb)

            # 2. Nets prediction.
            predicted_later_patch = forward_net(earlier_patch)
            predicted_earlier_patch = backward_net(later_patch)

            # 3. Losses.
            loss_later = 1 - cos_sim(predicted_later_patch, later_patch).mean()
            loss_earlier = 1 - cos_sim(predicted_earlier_patch, earlier_patch).mean()

            loss = loss_later + loss_earlier
            
            # 4. Logging.
            global_loss.append(loss.item())
            
            wandb.log({
                "train/batch_loss" : loss,
                })

            # 5. Optimization.
            if torch.isnan(loss) or torch.isinf(loss):
                exit()
            elif not torch.isnan(loss) and not torch.isinf(loss):
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
        # Global logging.
        wandb.log({
            "train/epoch_loss" : torch.Tensor(global_loss, device = 'cpu').mean(),
            })
        
    # Model saving.
    directory = f'{args.checkpoint_savepath}/{args.class_name}'

    if not os.path.exists(directory):
        os.makedirs(directory)

    torch.save(forward_net.state_dict(), os.path.join(directory, 'forward_net_' + model_name + '.pth'))
    torch.save(backward_net.state_dict(), os.path.join(directory, 'backward_net_' + model_name + '.pth'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Train L2BT framework on a dataset.')

    parser.add_argument('--dataset_path', default = './DATASETS/visa', type = str, 
                        help = 'Dataset path.')

    parser.add_argument('--checkpoint_savepath', default = './checkpoints/checkpoints_visa', type = str, 
                        help = 'Where to save the model checkpoints.')

    parser.add_argument('--class_name', default = "candle", type = str,
                        help = 'Category name.')

    parser.add_argument('--epochs_no', default = 50, type = int,
                        help = 'Number of epochs to train.')

    parser.add_argument('--img_size', default = 1036, type = int,
                        help = 'Square image resolution.')

    parser.add_argument('--batch_size', default = 4, type = int,
                        help = 'Batch dimension. Usually 16 is around the max.')

    parser.add_argument('--label', default = 'final_model', type = str, 
                        help = 'Label to identify the experiment.')

    args = parser.parse_args()
    train(args)