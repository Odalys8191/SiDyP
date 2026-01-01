import argparse
import torch
import random
#from rich.traceback import install
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import TensorDataset, SequentialSampler, DataLoader

from dataset import create_dataset
from utils import set_seed
from plc_finetune import PLC_Trainer
from knn import KNN_prior_dynamic
from simplex_diff_trainer import Simplex_Trainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--dataset", default="numclaim", type=str, help="dataset:[semeval, numclaim, chemprot, trec, 20news]")
    parser.add_argument("--noise_type", default="llm", type=str, help="label noise type:[llm, realworld, synthetic]")
    parser.add_argument("--llm_type", default="llama3-70b", type=str, help="llm model:[llama3-70b, gpt4o, mixtral822, llama31-70b, llama31-405b]")
    parser.add_argument("--prompt_type", default="zeroshot", type=str, help="llm prompting method:[zeroshot, fewshot]")
    parser.add_argument("--syn_type", default="SN", type=str, help="synthetic noise type:[SN, ASN, IDN]")
    # plc
    parser.add_argument("--plc", default="bert-base-uncased", type=str, help="pretrain language classifier model")
    parser.add_argument("--embed", default="WhereIsAI/UAE-Large-V1", type=str, help="embedding model for knn classifier")
    parser.add_argument("--train_batch_size", default=128, type=int, help="Batch size for training.")
    parser.add_argument("--eval_batch_size", default=128, type=int, help="Batch size for validation.")
    parser.add_argument('--alpha_t', type=float, default=5)
    parser.add_argument("--plc_epochs", default=10, type=int, help="Number of epochs for PLC training.")
    parser.add_argument("--plc_lr", default=5e-5, type=float, help="The initial learning rate for PLC Adam.")
    parser.add_argument("--num_model", type=float, default=3, help='The number of model branches')
    # dynamic prior retrieval
    parser.add_argument("--noise_ratio", type=float, default=0.3, help='The ratio of noisy data to be poisoned.')
    parser.add_argument("--K", type=int, default=10, help="certain label retrieval threshold")
    parser.add_argument("--certain_threshold", type=float, default=0.9, help="certain label retrieval threshold")
    parser.add_argument("--dominant_threshold", type=float, default=0.8, help="dominant label retrieval threshold")
    # diffusion
    parser.add_argument("--diff_epochs", type=int, default=10, help="training epochs for diffusion model")
    parser.add_argument("--warmup_epochs", default=0.2, help="warmup_epochs", type=float)
    parser.add_argument("--diff_lr", default=1e-3, type=float, help="The initial learning rate for diffusion Adam.")
    parser.add_argument("--train_timesteps", default=500, help="Number of training timesteps for diffusion model", type=int)
    parser.add_argument("--infer_timesteps", default=10, help="Number of inference timesteps for diffusion model", type=int)
    parser.add_argument("--num_sample", default=6, help="Number of sample for dynamic prior", type=int)
    parser.add_argument('--lambda_t', type=float, default=2)
    parser.add_argument("--beta_schedule", type=str, default="squaredcos_improved_ddpm")
    parser.add_argument("--simplex_value", type=float, default=5.0)
    parser.add_argument("--clip_sample", type=bool, default=False, help="Whether to clip predicted sample between -1 and 1 for numerical stability in the noise scheduler.")
    parser.add_argument("--ldl", action="store_true", help="Whether to use LDL (Label Distribution Learning) mode")
    parser.add_argument("--num_classes", type=int, default=None, help="Number of classes for LDL task")

    args = parser.parse_args()
    set_seed(args)
    device = torch.device(f'cuda' if torch.cuda.is_available() else "cpu")
    print(f'Device using: {device}')
    args.device = device

    if not args.ldl:
        if args.dataset.lower() == '20news':
            args.num_classes = 20
        elif args.dataset.lower() == 'numclaim':
            args.num_classes = 2
        elif args.dataset.lower() == 'trec':
            args.num_classes = 6
        elif args.dataset.lower() == 'semeval':
            args.num_classes = 9

        if args.noise_type == "llm":
            args.dataset_path = f"datasets/llm/{args.prompt_type}/{args.llm_type}"
        elif args.noise_type == "synthetic" or args.noise_type == "realworld":
            args.dataset_path = "datasets/realworld"
    else:
        # For LDL mode, use dataset as directory path for npy files
        args.dataset_path = args.dataset
        print(f"Using LDL mode with dataset path: {args.dataset_path}")

    print(args)

    import numpy as np
    from torch.utils.data import TensorDataset, SequentialSampler, DataLoader
    
    if not args.ldl:
        train_data, train_sampler, train_dataloader, train_embedding, valid_data, valid_sampler, \
            valid_dataloader, valid_embedding, test_data, test_sampler, test_dataloader, test_embedding = create_dataset(args)
        train_inputs = torch.stack([train_data[idx][0] for idx in range(len(train_data))], dim=0)
        train_masks = torch.stack([train_data[idx][1] for idx in range(len(train_data))], dim=0)
        train_true_labels = torch.stack([train_data[idx][2] for idx in range(len(train_data))], dim=0)
        train_noisy_labels = torch.tensor([train_data[idx][3] for idx in range(len(train_data))])
        valid_inputs = torch.stack([valid_data[idx][0] for idx in range(len(valid_data))], dim=0)
        valid_masks = torch.stack([valid_data[idx][1] for idx in range(len(valid_data))], dim=0)
        valid_true_labels = torch.stack([valid_data[idx][2] for idx in range(len(valid_data))], dim=0)
        valid_noisy_labels = torch.tensor([valid_data[idx][3] for idx in range(len(valid_data))])
        test_inputs = torch.stack([test_data[idx][0] for idx in range(len(test_data))], dim=0)
        test_masks = torch.stack([test_data[idx][1] for idx in range(len(test_data))], dim=0)
        test_true_labels = torch.stack([test_data[idx][2] for idx in range(len(test_data))], dim=0)
    else:
        # LDL模式下加载npy文件
        print("Loading LDL dataset...")
        
        # 加载特征和标签分布 - 先在CPU上加载
        train_features_orig = np.load(f"{args.dataset_path}/train_feature.npy")
        test_features = np.load(f"{args.dataset_path}/test_feature.npy")
        train_labels_orig = np.load(f"{args.dataset_path}/train_label.npy")
        test_labels = np.load(f"{args.dataset_path}/test_label.npy")
        
        # 确定类别数量
        if args.num_classes is None:
            args.num_classes = train_labels_orig.shape[1]
        
        # 分割训练集和验证集（支持随机划分） - 在CPU上进行
        # 使用随机种子确保可重复性
        indices = torch.randperm(len(train_features_orig), device='cpu')
        train_size = int(0.8 * len(train_features_orig))
        valid_size = len(train_features_orig) - train_size
        
        # 随机划分训练集和验证集
        train_indices = indices[:train_size]
        valid_indices = indices[train_size:]
        
        # 只将需要的数据转移到GPU，减少内存占用
        train_features = torch.tensor(train_features_orig[train_indices], dtype=torch.float32, device=args.device)
        valid_features = torch.tensor(train_features_orig[valid_indices], dtype=torch.float32, device=args.device)
        train_labels = torch.tensor(train_labels_orig[train_indices], dtype=torch.float32, device=args.device)
        valid_labels = torch.tensor(train_labels_orig[valid_indices], dtype=torch.float32, device=args.device)
        test_features = torch.tensor(test_features, dtype=torch.float32, device=args.device)
        test_labels = torch.tensor(test_labels, dtype=torch.float32, device=args.device)
        
        # 释放CPU内存
        del train_features_orig, train_labels_orig
        
        # 在LDL模式下，真实标签和噪声标签相同（标签分布）
        train_true_labels = train_labels
        train_noisy_labels = train_labels
        valid_true_labels = valid_labels
        valid_noisy_labels = valid_labels
        test_true_labels = test_labels
        
        # 为LDL模式创建必要的占位符张量
        # 需要核实：此处是否需要创建真实的masks和inputs，或仅使用占位符
        # 在LDL模式下，我们直接使用特征，不需要文本输入的masks
        train_inputs = torch.zeros((train_size, 1), device=args.device)  # 占位符，实际不使用
        train_masks = torch.ones((train_size, 1), device=args.device)    # 全1掩码，实际不使用
        valid_inputs = torch.zeros((valid_size, 1), device=args.device)  # 占位符，实际不使用
        valid_masks = torch.ones((valid_size, 1), device=args.device)    # 全1掩码，实际不使用
        test_inputs = torch.zeros((len(test_features), 1), device=args.device)  # 占位符，实际不使用
        test_masks = torch.ones((len(test_features), 1), device=args.device)    # 全1掩码，实际不使用
        
        # 创建数据集和数据加载器
        # 在LDL模式下，我们直接使用特征作为embedding
        train_embedding = train_features
        valid_embedding = valid_features
        test_embedding = test_features
        
        # 为PLC训练准备数据
        # 需要核实：PLC训练器是否能直接处理LDL特征，经修改PLC_Trainer可以直接处理特征输入
        train_data = TensorDataset(train_inputs, train_masks, train_true_labels, train_noisy_labels)
        train_sampler = SequentialSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)
        
        valid_data = TensorDataset(valid_inputs, valid_masks, valid_true_labels, valid_noisy_labels)
        valid_sampler = SequentialSampler(valid_data)
        valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=args.eval_batch_size)
        
        test_data = TensorDataset(test_inputs, test_masks, test_true_labels)
        test_sampler = SequentialSampler(test_data)
        test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=args.eval_batch_size)

    print("==========================Stage I: Pre-trained Language Classifier Finetuning===========================")
    
    if not args.ldl:
        # 原始模式下使用PLC_Trainer
        plc_trainer = PLC_Trainer(args, train_dataloader, valid_dataloader, test_dataloader)
        z_train, z_valid, z_test, best_plc_model, dists_list = plc_trainer.train()
    else:
        # LDL模式下，使用FeatureModel版本的PLC_Trainer处理特征输入
        print("LDL模式下使用FeatureModel版本的PLC_Trainer处理特征输入")
        
        # 为LDL模式创建专门的数据集和数据加载器
        # 训练集：(features, true_labels, noisy_labels)
        train_dataset = TensorDataset(train_features, train_true_labels, train_noisy_labels)
        train_sampler = SequentialSampler(train_dataset)
        train_dataloader_ldl = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)
        
        # 验证集：(features, true_labels, noisy_labels)
        valid_dataset = TensorDataset(valid_features, valid_true_labels, valid_noisy_labels)
        valid_sampler = SequentialSampler(valid_dataset)
        valid_dataloader_ldl = DataLoader(valid_dataset, sampler=valid_sampler, batch_size=args.eval_batch_size)
        
        # 测试集：(features, true_labels)
        test_dataset = TensorDataset(test_features, test_true_labels)
        test_sampler = SequentialSampler(test_dataset)
        test_dataloader_ldl = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.eval_batch_size)
        
        # 使用FeatureModel版本的PLC_Trainer
        feature_dim = train_features.shape[1]
        plc_trainer = PLC_Trainer(args, train_dataloader_ldl, valid_dataloader_ldl, test_dataloader_ldl, feature_dim)
        z_train, z_valid, z_test, best_plc_model, dists_list = plc_trainer.train()

    print("==========================Compute Training Dynamic Prior ==========================")
    print(z_train.shape) # (models, epochs, batch, dim)
    z_train = z_train.permute(2,0,1,3)
    B, M, N, D = z_train.shape
    z_train = z_train.reshape(B, M, N*D)
    z0_train = z_train[:, :, :D]

    z_valid = z_valid.permute(2,0,1,3)
    B2, M2, N2, D2 = z_valid.shape
    z_valid = z_valid.reshape(B2, M2, N2*D2)
    z0_valid = z_valid[:, :, :D2]

    z_test = z_test.permute(2,0,1,3)
    B3, M3, N3, D3 = z_test.shape
    z_test = z_test.reshape(B3, M3, N3*D3)
    z0_test = z_test[:, :, :D3]

    if not args.ldl:
        # Train noisy data detection
        dists_score_list = []
        markers_list = []
        for idx in range(dists_list.shape[0]):
            dists = dists_list[idx].squeeze()
            dists_labels = train_noisy_labels
            dists_mean = torch.mean(dists, 0)
            dists_mean = torch.tensor([dists_mean[i, dists_labels[i]] for i in range(len(dists_labels))])
            dists_var = torch.std(dists, 0)
            dists_var = torch.tensor([dists_var[i, dists_labels[i]] for i in range(len(dists_labels))])
            dists_score = dists_mean + dists_var
            dists_score = dists_score[:len(dists_labels)]
            markers = torch.zeros(len(dists_labels))
            number_points = int(len(dists_score) * args.noise_ratio)
            noisy_points = torch.topk(dists_score, number_points, largest=True).indices
            markers[noisy_points] = 1
            dists_score_list.append(dists_score.unsqueeze(0))
            markers_list.append(markers.unsqueeze(0))
        dists_score_list = torch.stack(dists_score_list, dim=0)
        markers_list = torch.stack(markers_list, dim=0)
    else:
        # LDL模式下，噪声检测逻辑不同，这里构造一个简单的markers_list
        # 需要核实：LDL模式下的噪声检测逻辑
        print("LDL模式下使用简化的噪声检测逻辑")
        markers_list = torch.zeros((M, B))  # 假设所有数据都是干净的
        
        # 对于验证集
        markers_list_valid = torch.zeros((M, B2))

    train_priors = []
    train_prior_weights = []
    train_uncertain_marker = []

    valid_priors = []

    for idx in range(M):
        if not args.ldl:
            knn_labels = train_noisy_labels
            knn_true_labels = train_true_labels
            # knn_z0 = z0_train[:, idx, :].squeeze()
            knn_z0 = train_embedding
            knn_prior = KNN_prior_dynamic(args, knn_z0, knn_labels, knn_true_labels, markers_list[idx].squeeze())
            priors, weights, uncertain_marker, true_labels = knn_prior.get_dynamic_prior(k=args.K)
        else:
            # LDL模式下，knn_labels和knn_true_labels是标签分布
            # 需要核实：KNN_prior_dynamic是否能处理标签分布
            knn_labels = torch.argmax(train_noisy_labels, dim=1)  # 暂时使用最大概率的类别作为标签
            knn_true_labels = torch.argmax(train_true_labels, dim=1)  # 暂时使用最大概率的类别作为真实标签
            knn_z0 = train_embedding
            knn_prior = KNN_prior_dynamic(args, knn_z0, knn_labels, knn_true_labels, markers_list[idx].squeeze())
            priors, weights, uncertain_marker, true_labels = knn_prior.get_dynamic_prior(k=args.K)
        
        train_priors.append(priors)
        train_prior_weights.append(weights)
        train_uncertain_marker.append(uncertain_marker)
    
    train_uncertain_marker = torch.stack(train_uncertain_marker, dim=0)

    if not args.ldl:
        dists_score_list = []
        markers_list = []
        dists_list = dists_list[:, :, B:(B+B2), :]
        for idx in range(dists_list.shape[0]):
            dists = dists_list[idx].squeeze()
            dists_labels = valid_noisy_labels
            dists_mean = torch.mean(dists, 0)
            dists_mean = torch.tensor([dists_mean[i, dists_labels[i]] for i in range(len(dists_labels))])
            dists_var = torch.std(dists, 0)
            dists_var = torch.tensor([dists_var[i, dists_labels[i]] for i in range(len(dists_labels))])
            dists_score = dists_mean + dists_var
            dists_score = dists_score[:len(dists_labels)]
            markers = torch.zeros(len(dists_labels))
            number_points = int(len(dists_score) * args.noise_ratio)
            noisy_points = torch.topk(dists_score, number_points, largest=True).indices
            markers[noisy_points] = 1
            dists_score_list.append(dists_score.unsqueeze(0))
            markers_list.append(markers.unsqueeze(0))
        dists_score_list = torch.stack(dists_score_list, dim=0)
        markers_list = torch.stack(markers_list, dim=0)


    for idx in range(M):
        if not args.ldl:
            knn_labels = valid_noisy_labels
            knn_true_labels = valid_true_labels
            knn_z0 = z0_valid[:, idx, :].squeeze()
            knn_z0 = valid_embedding
            knn_prior = KNN_prior_dynamic(args, knn_z0, knn_labels, knn_true_labels, markers_list[idx].squeeze())
            priors, weights, uncertain_marker, true_labels = knn_prior.get_dynamic_prior(k=args.K)

            valid_priors.append(torch.argmax(weights, dim=-1))
        else:
            # LDL模式下处理验证集
            knn_labels = torch.argmax(valid_noisy_labels, dim=1)  # 暂时使用最大概率的类别作为标签
            knn_true_labels = torch.argmax(valid_true_labels, dim=1)  # 暂时使用最大概率的类别作为真实标签
            knn_z0 = valid_embedding
            knn_prior = KNN_prior_dynamic(args, knn_z0, knn_labels, knn_true_labels, markers_list_valid[idx].squeeze())
            priors, weights, uncertain_marker, true_labels = knn_prior.get_dynamic_prior(k=args.K)

            valid_priors.append(torch.argmax(weights, dim=-1))
    
    
    # majority vote for multiple model branch valid priors
    valid_priors = torch.stack(valid_priors)
    valid_p, freq = torch.mode(valid_priors, dim=0)
    final_votes = torch.empty(valid_priors.size(-1), dtype=torch.long)
    for idx in range(valid_priors.size(-1)):
        valid_priors_model = valid_priors[:, idx]
        counts = valid_priors_model.bincount(minlength=valid_priors_model.max() + 1)
        max_freq = counts.max()
        tied = torch.where(counts == max_freq)[0]

        if len(tied) > 1:
            final_votes[idx] = random.choice(tied.tolist())
        else:
            final_votes[idx] = valid_p[idx]

    valid_priors = final_votes


    train_priors = pad_sequence([model.transpose(0,1) for model in train_priors], batch_first=True, padding_value=-1).transpose(1,2)

    train_prior_weights = torch.stack(train_prior_weights, dim=0)

    train_priors = train_priors.permute(1,0,2)
    train_prior_weights = train_prior_weights.permute(1,0,2)
    train_uncertain_marker = train_uncertain_marker.permute(1,0)

    scaler = torch.amp.GradScaler("cuda")

    # prepare datasets for generative model
    if not args.ldl:
        train_dataset = TensorDataset(z_train, train_priors, train_prior_weights, train_uncertain_marker, train_noisy_labels, train_true_labels, train_embedding)
    else:
        # LDL模式下，直接传递连续标签分布给Simplex_Trainer* （已经修改使其支持连续标签分布）
        train_dataset = TensorDataset(z_train, train_priors, train_prior_weights, train_uncertain_marker, train_noisy_labels, train_true_labels, train_embedding)
    
    if not args.ldl:
        valid_dataset = TensorDataset(valid_inputs, valid_masks, valid_priors, z_valid, valid_embedding)
    else:
        # LDL模式下，验证集包含真实标签分布
        valid_dataset = TensorDataset(valid_inputs, valid_masks, valid_true_labels, z_valid, valid_embedding)
    valid_sampler = SequentialSampler(valid_dataset)
    valid_dataloader = DataLoader(valid_dataset, sampler=valid_sampler, batch_size=args.eval_batch_size)

    if not args.ldl:
        test_dataset = TensorDataset(test_inputs, test_masks, test_true_labels, z_test, test_embedding)
    else:
        # LDL模式下，保留真实的连续标签分布
        test_dataset = TensorDataset(test_inputs, test_masks, test_true_labels, z_test, test_embedding)
    
    test_sampler = SequentialSampler(test_dataset)
    test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=args.eval_batch_size)

    simplex_trainer = Simplex_Trainer(args, train_dataset, valid_dataloader, test_dataloader, z_train.size(-1), best_plc_model)
    
    simplex_trainer.train()
if __name__ == "__main__": 
    # install(show_locals=False)
    main()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
