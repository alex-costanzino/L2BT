import argparse
import matplotlib.pyplot as plt
import os
import torch
import torchvision
from tqdm import tqdm
import numpy as np
import pandas as pd

from utils.loader_VisA import get_data_loader
from utils.general_utils import set_seeds

from models.teacher import FeatureExtractor
from models.students import FeatureProjectionMLP

from utils.metrics_utils import calculate_au_pro
from sklearn.metrics import roc_auc_score

from utils.general_utils import set_seeds, imagenet_denormalize

def infer(args):

    set_seeds()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Dataloader.
    test_loader, max_hw = get_data_loader(
        "test", class_name = args.class_name,
        img_size = args.img_size,
        dataset_path = args.dataset_path)

    # Feature extractor.
    fe = FeatureExtractor(layers = [7,11]).to(device).eval() 

    # Model instantiation.
    backward_net = FeatureProjectionMLP(in_features = fe.embed_dim, out_features = fe.embed_dim).to(device)
    forward_net = FeatureProjectionMLP(in_features = fe.embed_dim, out_features = fe.embed_dim).to(device)

    forward_net_path = rf'{args.checkpoint_folder}/{args.class_name}/forward_net_{args.label}_{args.class_name}_{args.epochs_no}ep_{args.batch_size}bs.pth'
    backward_net_path = rf'{args.checkpoint_folder}/{args.class_name}/backward_net_{args.label}_{args.class_name}_{args.epochs_no}ep_{args.batch_size}bs.pth'

    forward_net.load_state_dict(torch.load(forward_net_path, weights_only=False))
    backward_net.load_state_dict(torch.load(backward_net_path, weights_only=False))

    # Send students to GPU.
    forward_net.to(device)
    backward_net.to(device)

    # Make students non-trainable.
    forward_net.eval()
    backward_net.eval()

    # Gaussian blur parameters.
    w_l, w_u = 5, 7
    pad_l, pad_u = 2, 3
    weight_l = torch.ones(1, 1, w_l, w_l, device=device)/(w_l**2)
    weight_u = torch.ones(1, 1, w_u, w_u, device=device)/(w_u**2)
    
    predictions, gts = [], []
    image_labels, pixel_labels = [], []
    image_preds, pixel_preds = [], []

    inference_time = []

    start_event = torch.cuda.Event(enable_timing = True)
    end_event = torch.cuda.Event(enable_timing = True)

    for rgb, gt, label, rgb_path in tqdm(test_loader, desc = f'Extracting features from class: {args.class_name}.'):
        
        defect_class_str = rgb_path[0].split('/')[-3]
        image_name_str = rgb_path[0].split('/')[-1]

        with torch.no_grad():

            rgb = rgb.to(device)

            start_event.record()

            # 1. Feature extraction.
            last_patch, middle_patch = fe(rgb)

            # 4. Nets prediction.
            predicted_middle_patch = backward_net(last_patch)
            predicted_last_patch = forward_net(middle_patch)

            middle_anomaly_map = (torch.nn.functional.normalize(predicted_middle_patch, dim = -1) - torch.nn.functional.normalize(middle_patch, dim = -1)).pow(2).sum(-1).sqrt()
            last_anomaly_map = (torch.nn.functional.normalize(predicted_last_patch, dim = -1) - torch.nn.functional.normalize(last_patch, dim = -1)).pow(2).sum(-1).sqrt()

            combined_anomaly_map = (middle_anomaly_map * last_anomaly_map).reshape(1, 1, args.img_size // fe.patch_size, args.img_size // fe.patch_size)

            # Upsample to original resolution.
            combined_anomaly_map = torch.nn.functional.interpolate(combined_anomaly_map, size = [max_hw, max_hw], mode = 'bilinear')

            # Approximated Gaussian blur.
            combined_anomaly_map = torch.nn.functional.conv2d(input = combined_anomaly_map, padding = pad_l, weight = weight_l)
            combined_anomaly_map = torch.nn.functional.conv2d(input = combined_anomaly_map, padding = pad_l, weight = weight_l)
            combined_anomaly_map = torch.nn.functional.conv2d(input = combined_anomaly_map, padding = pad_l, weight = weight_l)
            combined_anomaly_map = torch.nn.functional.conv2d(input = combined_anomaly_map, padding = pad_l, weight = weight_l)
            combined_anomaly_map = torch.nn.functional.conv2d(input = combined_anomaly_map, padding = pad_l, weight = weight_l)

            combined_anomaly_map = torch.nn.functional.conv2d(input = combined_anomaly_map, padding = pad_u, weight = weight_u)
            combined_anomaly_map = torch.nn.functional.conv2d(input = combined_anomaly_map, padding = pad_u, weight = weight_u)
            combined_anomaly_map = torch.nn.functional.conv2d(input = combined_anomaly_map, padding = pad_u, weight = weight_u)

            combined_anomaly_map = combined_anomaly_map.reshape(max_hw, max_hw)

            combined_anomaly_map = torchvision.transforms.functional.center_crop(combined_anomaly_map, gt.shape[-2:])

            end_event.record()
            torch.cuda.synchronize()
            inf_time = start_event.elapsed_time(end_event)

            inference_time.append(inf_time)

            # 7. Prediction and ground-truth accumulation.
            gts.append(gt.squeeze().cpu().detach().numpy())
            predictions.append((combined_anomaly_map).cpu().detach().numpy())

            # GTs.
            image_labels.append(label)
            pixel_labels.extend(gt.flatten().cpu().detach().numpy()) 

            # Predictions.
            K = int((combined_anomaly_map.shape[0] * combined_anomaly_map.shape[1]) * 0.001)
            global_score = torch.topk(combined_anomaly_map.flatten(), k=K)[0].mean().cpu().detach().numpy()

            image_preds.append(global_score)
            pixel_preds.extend(combined_anomaly_map.flatten().cpu().detach().numpy()) 

            if args.produce_qualitatives:

                save_path = f'{args.qualitative_folder}/{args.label}_{args.class_name}_{args.epochs_no}ep_{args.batch_size}bs/{defect_class_str}'

                if not os.path.exists(save_path):
                    os.makedirs(save_path)

                _, axs = plt.subplots(1,3, figsize = (7,3))

                rgb = imagenet_denormalize(rgb)
                rgb = torch.nn.functional.interpolate(rgb, size = [max_hw, max_hw], mode = 'bilinear')
                rgb = torchvision.transforms.functional.center_crop(rgb, gt.shape[-2:])
                axs[0].imshow(rgb.squeeze().permute(1,2,0).cpu().detach().numpy())
                axs[0].set_title('Input Image')

                axs[1].imshow(gt.squeeze().cpu().detach().numpy(), cmap='gray')
                axs[1].set_title('Ground-truth')

                axs[2].imshow(combined_anomaly_map.cpu().detach().numpy(), cmap=plt.cm.jet)
                axs[2].set_title('Anomaly Map')

                # Remove ticks and labels from all subplots.
                for ax in axs.flat:
                    ax.set_xticks([])
                    ax.set_yticks([])
                    ax.set_xticklabels([])
                    ax.set_yticklabels([])

                # Adjust the layout and spacing.
                plt.tight_layout()

                plt.savefig(os.path.join(save_path, image_name_str), dpi = 256)

                if args.visualize_plot:
                    plt.show()

                plt.close()

    # Calculate AD&S metrics.
    au_pros_q4, _, weights = calculate_au_pro(gts, predictions, weighted = False)

    q1, q2, q3 = np.quantile(weights, 0.25), np.quantile(weights, 0.5), np.quantile(weights, 0.75)

    au_pros_q3, _, _ = calculate_au_pro(gts, predictions, weighted = False, size_thr = (0, q3))
    au_pros_q2, _, _ = calculate_au_pro(gts, predictions, weighted = False, size_thr = (0, q2))
    au_pros_q1, _, _ = calculate_au_pro(gts, predictions, weighted = False, size_thr = (0, q1))

    pixel_rocauc = roc_auc_score(np.array(pixel_labels),  np.array(pixel_preds))
    image_rocauc = roc_auc_score(np.stack(image_labels), np.stack(image_preds))

    result_file_name = f'{args.quantitative_folder}/{args.label}_{args.class_name}_{args.epochs_no}ep_{args.batch_size}bs.md'

    title_string = f'Metrics for class {args.class_name} with {args.epochs_no}ep_{args.batch_size}bs'
    type_string_c = 'Cumulative'
    header_string = 'AUPRO@30% & AUPRO@10% & AUPRO@5% & AUPRO@1% & P-AUROC & I-AUROC'
   
    results_string_q4 = f'{au_pros_q4[0]:.3f} & {au_pros_q4[1]:.3f} & {au_pros_q4[2]:.3f} & {au_pros_q4[3]:.3f} & {pixel_rocauc:.3f} & {image_rocauc:.3f}'
    results_string_q3 = f'{au_pros_q3[0]:.3f} & {au_pros_q3[1]:.3f} & {au_pros_q3[2]:.3f} & {au_pros_q3[3]:.3f}'
    results_string_q2 = f'{au_pros_q2[0]:.3f} & {au_pros_q2[1]:.3f} & {au_pros_q2[2]:.3f} & {au_pros_q2[3]:.3f}'
    results_string_q1 = f'{au_pros_q1[0]:.3f} & {au_pros_q1[1]:.3f} & {au_pros_q1[2]:.3f} & {au_pros_q1[3]:.3f}'

    if not os.path.exists(args.quantitative_folder):
        os.makedirs(args.quantitative_folder)

    with open(result_file_name, "w") as markdown_file:
        markdown_file.write(
            title_string + '\n' + 
            type_string_c + '\n' + 
            header_string + '\n' + 
            results_string_q4 + '\n' + 
            results_string_q3 + '\n' + 
            results_string_q2 + '\n' + 
            results_string_q1 + '\n'
            )

    # Print AD&S metrics.
    print(title_string)
    print("P-AUROC  |  I-AUROC")
    print(f'  {pixel_rocauc:.3f}   |   {image_rocauc:.3f}', end = '\n')
    
    print(" QUARTILE | AUPRO@30% | AUPRO@10% | AUPRO@5% | AUPRO@1% |")
    print(f'    Q4    |   {au_pros_q4[0]:.3f}   |   {au_pros_q4[1]:.3f}   |   {au_pros_q4[2]:.3f}  |   {au_pros_q4[3]:.3f}  |', end = '\n')
    print(f'    Q3    |   {au_pros_q3[0]:.3f}   |   {au_pros_q3[1]:.3f}   |   {au_pros_q3[2]:.3f}  |   {au_pros_q3[3]:.3f}  |', end = '\n')
    print(f'    Q2    |   {au_pros_q2[0]:.3f}   |   {au_pros_q2[1]:.3f}   |   {au_pros_q2[2]:.3f}  |   {au_pros_q2[3]:.3f}  |', end = '\n')
    print(f'    Q1    |   {au_pros_q1[0]:.3f}   |   {au_pros_q1[1]:.3f}   |   {au_pros_q1[2]:.3f}  |   {au_pros_q1[3]:.3f}  |', end = '\n')

    results = {

        'class_name' : [], 'inference_time' : [], 'i_auroc' : [], 'p_auroc' : [],

        'aupro_30_q4' : [], 'aupro_10_q4' : [], 'aupro_05_q4' : [], 'aupro_01_q4' : [],
        'aupro_30_q3' : [], 'aupro_10_q3' : [], 'aupro_05_q3' : [], 'aupro_01_q3' : [],
        'aupro_30_q2' : [], 'aupro_10_q2' : [], 'aupro_05_q2' : [], 'aupro_01_q2' : [],
        'aupro_30_q1' : [], 'aupro_10_q1' : [], 'aupro_05_q1' : [], 'aupro_01_q1' : [],

        }
    
    # Classic.
    results['class_name'].append(args.class_name)
    results['inference_time'].append(np.mean(inference_time))
    results['i_auroc'].append(image_rocauc)
    results['p_auroc'].append(pixel_rocauc)
    
    # Cumulatives.
    results['aupro_30_q4'].append(au_pros_q4[0])
    results['aupro_10_q4'].append(au_pros_q4[1])
    results['aupro_05_q4'].append(au_pros_q4[2])
    results['aupro_01_q4'].append(au_pros_q4[3])

    results['aupro_30_q3'].append(au_pros_q3[0])
    results['aupro_10_q3'].append(au_pros_q3[1])
    results['aupro_05_q3'].append(au_pros_q3[2])
    results['aupro_01_q3'].append(au_pros_q3[3])

    results['aupro_30_q2'].append(au_pros_q2[0])
    results['aupro_10_q2'].append(au_pros_q2[1])
    results['aupro_05_q2'].append(au_pros_q2[2])
    results['aupro_01_q2'].append(au_pros_q2[3])

    results['aupro_30_q1'].append(au_pros_q1[0])
    results['aupro_10_q1'].append(au_pros_q1[1])
    results['aupro_05_q1'].append(au_pros_q1[2])
    results['aupro_01_q1'].append(au_pros_q1[3])

    results_df = pd.DataFrame(results)
    results_df.to_csv(result_file_name.replace('md', 'csv'), index=False, sep='&')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = 'Infer with L2BT framework on a dataset.')

    parser.add_argument('--checkpoint_folder', default = './checkpoints/checkpoints_visa', type = str,
                        help = 'Path to the folder containing students checkpoints.')

    parser.add_argument('--qualitative_folder', default = './results/qualitatives_visa', type = str,
                        help = 'Path to the folder in which to save the qualitatives.')

    parser.add_argument('--quantitative_folder', default = './results/quantitatives_visa', type = str,
                        help = 'Path to the folder in which to save the quantitatives.')

    parser.add_argument('--visualize_plot', default = False, action = 'store_true',
                        help = 'Whether to show plot or not.')

    parser.add_argument('--produce_qualitatives', default = True, action = 'store_true',
                        help = 'Whether to produce qualitatives or not.')

    parser.add_argument('--dataset_path', default = '/home/alex/research/anomaly_detection/DATASETS/visa', type = str,
                        help = 'Dataset path.')

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
    infer(args)