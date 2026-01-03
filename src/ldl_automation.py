#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LDL任务自动化评测脚本

功能：
1. 在run_0数据集上执行网格搜索
2. 使用最优超参数在所有run_i上运行实验
3. 生成性能报告和可视化结果

使用方法：
python ldl_automation.py --dataset_root {数据集根目录} --num_classes {类别数} [--config {配置文件路径}]
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from itertools import product
from typing import Dict, List, Tuple

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 直接导入main.py中的函数
from main import run_experiment, run_cv_experiment
from main import argparse as main_argparse
import torch
from torch.utils.data import TensorDataset, SequentialSampler, DataLoader

# 实现平均改进率计算函数
def calc_avg_imp(our_mean, sota_vals):
    """计算平均改进率"""
    if sota_vals.size == 0:
        return 0.0
    
    # 确保所有数据都是数值类型
    our_mean = np.asarray(our_mean, dtype=np.float64)
    sota_vals = np.asarray(sota_vals, dtype=np.float64)
    
    imps = []
    # 确保索引不超出数组长度
    min_length = min(len(our_mean), len(sota_vals), 6)
    
    for i in range(min_length):
        if i < 4:  # 前4个越小越好
            imps.append((sota_vals[i] - our_mean[i]) / (sota_vals[i] + 1e-12))
        else:  # 后2个越大越好
            imps.append((our_mean[i] - sota_vals[i]) / (sota_vals[i] + 1e-12))
    
    return np.mean(imps) if imps else 0.0


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="LDL任务自动化评测脚本")
    parser.add_argument("--dataset_root", required=True, type=str, help="数据集根目录路径")
    parser.add_argument("--num_classes", required=True, type=int, help="类别数量")
    parser.add_argument("--config", default=None, type=str, help="配置文件路径")
    parser.add_argument("--grid_search", action="store_true", help="仅执行网格搜索")
    parser.add_argument("--evaluate", action="store_true", help="仅执行模型评估")
    parser.add_argument("--seed", default=0, type=int, help="基础随机种子")
    parser.add_argument("--output_dir", default="./ldl_results", type=str, help="结果输出目录")
    parser.add_argument("--sota_json", default=None, type=str, help="Path to SOTA JSON file")
    return parser.parse_args()


class LDLAutomation:
    """LDL任务自动化评测类"""
    
    def __init__(self, args):
        self.args = args
        self.dataset_root = args.dataset_root
        self.num_classes = args.num_classes
        self.base_seed = args.seed
        
        # 从dataset_root中提取数据集名称
        self.dataset_name = os.path.basename(self.dataset_root)
        
        # 按数据集名称组织输出目录
        self.output_dir = os.path.join(args.output_dir, self.dataset_name)
        
        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 超参数搜索空间（仅搜索diff_lr和train_batch_size）
        self.hyperparam_space = {
            'diff_lr': [1e-3, 1e-2],
            'train_batch_size': [128, 256]
        }
        
        # 阶段配置参数
        # 搜索阶段：使用较小的epoch值加速搜索过程
        self.search_plc_epochs = 10  # 搜索阶段的PLC训练轮数
        self.search_diff_epochs = 10  # 搜索阶段的Diffusion训练轮数
        
        # 正式阶段：使用较大的epoch值确保模型充分训练
        self.formal_plc_epochs = 100  # 正式阶段的PLC训练轮数
        self.formal_diff_epochs = 100  # 正式阶段的Diffusion训练轮数
        
        # 固定超参数（使用main.py中的默认值）
        self.fixed_params = {
            'seed': self.base_seed,
            'num_classes': self.num_classes,
            'ldl': True,
            'plc_lr': 5e-5,
            'eval_batch_size': 128,
            'plc_epochs': self.formal_plc_epochs,  # 默认使用正式阶段的epoch值
            'diff_epochs': self.formal_diff_epochs,  # 默认使用正式阶段的epoch值
            'sota_json': self.args.sota_json
        }
        
        # 加载SOTA数据
        self.sota_data = self._load_sota_data()
        
        # 记录实验结果
        self.grid_search_results = []
        self.eval_results = []
        
        # 记录最优模型信息
        self.best_models = []
    
    def _load_sota_data(self) -> Dict:
        """加载SOTA数据"""
        # 使用命令行参数传递的sota_json路径
        sota_path = self.args.sota_json
        if sota_path and os.path.exists(sota_path):
            with open(sota_path, 'r') as f:
                return json.load(f)
        elif sota_path and not os.path.exists(sota_path):
            print(f"警告：SOTA数据文件 {sota_path} 不存在")
            return {}
        else:
            print("未指定SOTA JSON文件路径，将不使用SOTA数据进行比较")
            return {}
    
    def _get_run_path(self, run_idx: int) -> str:
        """获取run_i的路径"""
        return os.path.join(self.dataset_root, f"run_{run_idx}")
    
    def _prepare_data_for_run(self, run_idx: int, params: Dict):
        """为单个run准备数据"""
        import numpy as np
        from main import create_dataset
        
        # 创建main.py所需的args对象
        main_args = type('Args', (), {})
        
        # 设置所有参数
        for key, value in params.items():
            setattr(main_args, key, value)
        
        # 设置设备
        main_args.device = torch.device(f'cuda' if torch.cuda.is_available() else "cpu")
        
        # 加载数据
        run_path = self._get_run_path(run_idx)
        main_args.dataset_path = run_path
        
        # 加载特征和标签分布 - 先在CPU上加载
        train_features_orig = np.load(f"{run_path}/train_feature.npy")
        test_features = np.load(f"{run_path}/test_feature.npy")
        train_labels_orig = np.load(f"{run_path}/train_label.npy")
        test_labels = np.load(f"{run_path}/test_label.npy")
        
        # 确定类别数量
        if main_args.num_classes is None:
            main_args.num_classes = train_labels_orig.shape[1]
        
        # 分割训练集和验证集（8:2比例） - 在CPU上进行
        indices = torch.randperm(len(train_features_orig), device='cpu')
        train_size = int(0.8 * len(train_features_orig))
        valid_size = len(train_features_orig) - train_size
        
        # 随机划分训练集和验证集
        train_indices = indices[:train_size]
        valid_indices = indices[train_size:]
        
        # 只将需要的数据转移到GPU，减少内存占用
        train_features = torch.tensor(train_features_orig[train_indices], dtype=torch.float32, device=main_args.device)
        valid_features = torch.tensor(train_features_orig[valid_indices], dtype=torch.float32, device=main_args.device)
        train_labels = torch.tensor(train_labels_orig[train_indices], dtype=torch.float32, device=main_args.device)
        valid_labels = torch.tensor(train_labels_orig[valid_indices], dtype=torch.float32, device=main_args.device)
        test_features = torch.tensor(test_features, dtype=torch.float32, device=main_args.device)
        test_labels = torch.tensor(test_labels, dtype=torch.float32, device=main_args.device)
        
        # 在LDL模式下，真实标签和噪声标签相同（标签分布）
        train_true_labels = train_labels
        train_noisy_labels = train_labels
        valid_true_labels = valid_labels
        valid_noisy_labels = valid_labels
        test_true_labels = test_labels
        
        # 为LDL模式创建必要的占位符张量
        train_inputs = torch.zeros((train_size, 1), device=main_args.device)  # 占位符，实际不使用
        train_masks = torch.ones((train_size, 1), device=main_args.device)    # 全1掩码，实际不使用
        valid_inputs = torch.zeros((valid_size, 1), device=main_args.device)  # 占位符，实际不使用
        valid_masks = torch.ones((valid_size, 1), device=main_args.device)    # 全1掩码，实际不使用
        test_inputs = torch.zeros((len(test_features), 1), device=main_args.device)  # 占位符，实际不使用
        test_masks = torch.ones((len(test_features), 1), device=main_args.device)    # 全1掩码，实际不使用
        
        # 创建数据集和数据加载器
        # 在LDL模式下，我们直接使用特征作为embedding
        train_embedding = train_features
        valid_embedding = valid_features
        test_embedding = test_features
        
        # 为PLC训练准备数据
        train_data = TensorDataset(train_inputs, train_masks, train_true_labels, train_noisy_labels)
        train_sampler = SequentialSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=main_args.train_batch_size)
        
        valid_data = TensorDataset(valid_inputs, valid_masks, valid_true_labels, valid_noisy_labels)
        valid_sampler = SequentialSampler(valid_data)
        valid_dataloader = DataLoader(valid_data, sampler=valid_sampler, batch_size=main_args.eval_batch_size)
        
        test_data = TensorDataset(test_inputs, test_masks, test_true_labels)
        test_sampler = SequentialSampler(test_data)
        test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=main_args.eval_batch_size)
        
        return main_args, train_data, train_sampler, train_dataloader, train_embedding, \
               valid_data, valid_sampler, valid_dataloader, valid_embedding, \
               test_data, test_sampler, test_dataloader, test_embedding, \
               train_inputs, train_masks, train_true_labels, train_noisy_labels, \
               valid_inputs, valid_masks, valid_true_labels, valid_noisy_labels, \
               test_inputs, test_masks, test_true_labels
    
    def _run_experiment(self, params: Dict, run_idx: int = 0, use_cv: bool = True) -> Tuple[Dict, float]:
        """运行单次实验
        
        Args:
            params: 超参数字典
            run_idx: 运行索引
            use_cv: 是否执行五折交叉验证，默认为True
            
        Returns:
            exp_result: 实验结果
            imp: 改进率
        """
        print(f"运行实验: run_{run_idx}, 参数: {params}, 使用五折交叉验证: {use_cv}")
        
        # 按run_idx组织实验结果目录
        run_dir = os.path.join(self.output_dir, f"run_{run_idx}")
        os.makedirs(run_dir, exist_ok=True)
        
        # 创建main.py所需的args对象
        main_args = type('Args', (), {})
        
        # 设置所有参数
        for key, value in params.items():
            setattr(main_args, key, value)
        
        # 设置设备
        main_args.device = torch.device(f'cuda' if torch.cuda.is_available() else "cpu")
        
        # 加载数据
        run_path = self._get_run_path(run_idx)
        main_args.dataset_path = run_path
        
        # 加载特征和标签分布 - 先在CPU上加载
        train_features_orig = np.load(f"{run_path}/train_feature.npy")
        test_features = np.load(f"{run_path}/test_feature.npy")
        train_labels_orig = np.load(f"{run_path}/train_label.npy")
        test_labels = np.load(f"{run_path}/test_label.npy")
        
        # 确定类别数量
        if main_args.num_classes is None:
            main_args.num_classes = train_labels_orig.shape[1]
        
        metrics = {}
        
        if use_cv:
            # 执行五折交叉验证
            print("执行5折交叉验证...")
            main_args.cv_folds = 5  # 固定为5折交叉验证
            main_args.train_ratio = 0.6  # 固定为60%训练集，40%验证集
            
            cv_results = run_cv_experiment(main_args, train_features_orig, train_labels_orig, test_features, test_labels)
            
            # 收集所有折的metrics
            all_metrics = []
            for fold_result in cv_results:
                if 'metrics' in fold_result:
                    all_metrics.append(fold_result['metrics'])
            
            # 计算平均metrics
            if all_metrics:
                metrics_df = pd.DataFrame(all_metrics)
                avg_metrics = metrics_df.mean().to_dict()
                metrics = avg_metrics
        else:
            # 不执行五折交叉验证，直接运行实验
            print("直接运行实验（不执行五折交叉验证）...")
            
            # 准备数据，使用_run_prepare_data_for_run方法
            main_args, train_data, train_sampler, train_dataloader, train_embedding, \
            valid_data, valid_sampler, valid_dataloader, valid_embedding, \
            test_data, test_sampler, test_dataloader, test_embedding, \
            train_inputs, train_masks, train_true_labels, train_noisy_labels, \
            valid_inputs, valid_masks, valid_true_labels, valid_noisy_labels, \
            test_inputs, test_masks, test_true_labels = self._prepare_data_for_run(run_idx, params)
            
            # 直接调用run_experiment函数
            result = run_experiment(main_args, train_data, train_sampler, train_dataloader, train_embedding, \
                                   valid_data, valid_sampler, valid_dataloader, valid_embedding, \
                                   test_data, test_sampler, test_dataloader, test_embedding, \
                                   train_inputs, train_masks, train_true_labels, train_noisy_labels, \
                                   valid_inputs, valid_masks, valid_true_labels, valid_noisy_labels, \
                                   test_inputs, test_masks, test_true_labels)
            
            # 获取metrics
            if result and 'metrics' in result:
                metrics = result['metrics']
        
        # 计算改进率
        our_metrics = []
        if metrics:
            # 按顺序提取六个指标
            metric_order = ['chebyshev', 'clark', 'canberra', 'kl_div', 'cosine_similarity', 'intersection']
            for metric in metric_order:
                if metric in metrics:
                    our_metrics.append(metrics[metric])
        
        # 确保有6个指标值，如果不足则填充
        our_mean = our_metrics[:6] + [1.0]*(6 - len(our_metrics))
        
        # 处理sota_vals，根据当前数据集获取正确的指标数据
        sota_values = []
        if self.sota_data and 'data' in self.sota_data:
            # 检查当前数据集是否在sota数据中
            if self.dataset_name in self.sota_data['data']:
                dataset_sota = self.sota_data['data'][self.dataset_name]
                # 定义指标顺序，确保与our_mean的顺序一致
                metric_order = ['Cheby', 'Clark', 'Canbe', 'KL', 'Cosine', 'Inter']
                # 提取每个指标的mean值
                for metric in metric_order:
                    if metric in dataset_sota and 'mean' in dataset_sota[metric]:
                        sota_values.append(dataset_sota[metric]['mean'])
        
        # 确保sota_vals有6个值，如果不足则用1.0填充
        sota_vals = sota_values[:6] + [1.0]*(6 - len(sota_values))
        imp = calc_avg_imp(np.array(our_mean), np.array(sota_vals))
        
        # 获取模型保存路径
        # 模型保存路径格式：best_models/ldl/数据集名称/种子值
        model_save_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "best_models", "ldl", self.dataset_name, str(params['seed'])
        )
        
        # 保存实验配置和结果
        # 添加阶段标识，区分不同阶段的结果
        stage = "grid_search" if use_cv else "formal_experiment"
        
        exp_result = {
            'exp_id': f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            'params': params,
            'run_idx': run_idx,
            'metrics': metrics,
            'improvement': imp,
            'exp_dir': run_dir,
            'model_save_dir': model_save_dir,
            'use_cv': use_cv,
            'stage': stage
        }
        
        return exp_result, imp
    
    def grid_search(self) -> Dict:
        """执行网格搜索
        
        Returns:
            best_params: 最优超参数组合
        """
        print("开始网格搜索...")
        print(f"搜索阶段配置：plc_epochs={self.search_plc_epochs}, diff_epochs={self.search_diff_epochs}")
        
        # 生成所有超参数组合
        hyperparam_combinations = list(product(*self.hyperparam_space.values()))
        hyperparam_names = list(self.hyperparam_space.keys())
        
        best_imp = -float('inf')
        best_params = None
        best_run_results = []
        
        # 执行网格搜索
        for i, combo in enumerate(hyperparam_combinations):
            # 构建超参数字典
            params = dict(zip(hyperparam_names, combo))
            params.update(self.fixed_params)
            
            # 设置搜索阶段的小epoch值，加速搜索过程
            params['plc_epochs'] = self.search_plc_epochs
            params['diff_epochs'] = self.search_diff_epochs
            
            print(f"\n正在测试超参数组合 {i+1}/{len(hyperparam_combinations)}: {params}")
            
            # 执行单次实验，使用base_seed作为随机种子
            exp_result, imp = self._run_experiment(params, run_idx=0, use_cv=True)
            
            # 保存结果
            run_results = [exp_result]
            self.grid_search_results.append(exp_result)
            
            print(f"实验改进率: {imp:.4f}")
            
            # 更新最优超参数，以"sota改进率"作为核心评价指标
            if imp > best_imp:
                best_imp = imp
                best_params = params.copy()
                best_run_results = run_results
                print(f"找到更优超参数组合，改进率: {best_imp:.4f}")
        
        # 保存网格搜索结果
        self._save_grid_search_results()
        
        # 保存最优模型
        self._save_best_models(best_run_results, "grid_search")
        
        print(f"\n网格搜索完成！")
        print(f"最优超参数组合: {best_params}")
        print(f"最优改进率: {best_imp:.4f}")
        
        return best_params
    
    def _save_best_models(self, run_results: List[Dict], stage: str):
        """保存最优模型到指定位置"""
        import shutil
        
        for run_idx, result in enumerate(run_results):
            # 目标目录：数据集名称/run_i
            target_dir = os.path.join(self.output_dir, f"run_{run_idx}")
            os.makedirs(target_dir, exist_ok=True)
            
            # 检查模型目录是否存在
            model_dir = result['model_save_dir']
            if os.path.exists(model_dir):
                # 复制模型文件到目标目录
                for root, dirs, files in os.walk(model_dir):
                    for file in files:
                        src_file = os.path.join(root, file)
                        dst_file = os.path.join(target_dir, file)
                        shutil.copy2(src_file, dst_file)
                print(f"已将最优模型从 {model_dir} 复制到 {target_dir}")
            else:
                print(f"警告：模型目录 {model_dir} 不存在，无法复制模型")
        
        print(f"{stage}阶段最优模型已保存到指定位置")
    
    def evaluate_best_params(self, best_params: Dict) -> List[Dict]:
        """使用最优超参数在所有run_i上评估
        
        Args:
            best_params: 最优超参数组合
            
        Returns:
            eval_results: 评估结果列表
        """
        print("\n开始使用最优超参数评估所有run_i...")
        print(f"正式实验阶段配置：plc_epochs={self.formal_plc_epochs}, diff_epochs={self.formal_diff_epochs}")
        
        eval_results = []
        
        # 使用最优超参数，修改epoch相关参数为正式阶段的大值
        # 保持其他超参数不变，确保两阶段之间的超参数传递正确
        eval_params = best_params.copy()
        eval_params['plc_epochs'] = self.formal_plc_epochs
        eval_params['diff_epochs'] = self.formal_diff_epochs
        
        print(f"使用的最优超参数组合: {eval_params}")
        
        # 针对每个run_i（包括run_0）运行实验
        for run_idx in range(10):
            print(f"\n正在评估 run_{run_idx}...")
            
            # 使用不同的随机种子
            params = eval_params.copy()
            params['seed'] = self.base_seed + run_idx
            
            # 执行实验，不使用五折交叉验证
            exp_result, imp = self._run_experiment(params, run_idx=run_idx, use_cv=False)
            eval_results.append(exp_result)
            self.eval_results.append(exp_result)
            
            print(f"run_{run_idx} 改进率: {imp:.4f}")
        
        # 保存评估结果
        self._save_eval_results()
        
        # 保存最优模型
        self._save_best_models(eval_results, "evaluation")
        
        return eval_results
    
    def _save_grid_search_results(self):
        """保存网格搜索结果"""
        # 为搜索阶段创建专门的结果目录
        search_results_dir = os.path.join(self.output_dir, "grid_search")
        os.makedirs(search_results_dir, exist_ok=True)
        
        result_path = os.path.join(search_results_dir, "grid_search_results.json")
        with open(result_path, 'w') as f:
            json.dump(self.grid_search_results, f, indent=2)
        print(f"网格搜索结果已保存至 {result_path}")
    
    def _save_eval_results(self):
        """保存评估结果"""
        # 为正式实验阶段创建专门的结果目录
        eval_results_dir = os.path.join(self.output_dir, "formal_experiment")
        os.makedirs(eval_results_dir, exist_ok=True)
        
        result_path = os.path.join(eval_results_dir, "eval_results.json")
        with open(result_path, 'w') as f:
            json.dump(self.eval_results, f, indent=2)
        print(f"评估结果已保存至 {result_path}")
    
    def generate_report(self):
        """生成性能报告"""
        print("\n生成性能报告...")
        
        # 提取指标
        metrics = []
        for result in self.eval_results:
            metrics.append(result['metrics'])
        
        # 计算平均值和标准差
        metrics_df = pd.DataFrame(metrics)
        mean_metrics = metrics_df.mean().to_dict()
        std_metrics = metrics_df.std().to_dict()
        
        # 生成报告
        report = {
            'dataset': self.dataset_root,
            'num_classes': self.num_classes,
            'base_seed': self.base_seed,
            'evaluation_time': datetime.now().isoformat(),
            'mean_metrics': mean_metrics,
            'std_metrics': std_metrics,
            'sota_comparison': self.sota_data,
            'detailed_results': self.eval_results
        }
        
        # 保存报告
        report_path = os.path.join(self.output_dir, "performance_report.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # 生成可视化
        self._generate_visualizations(metrics_df, mean_metrics, std_metrics)
        
        print(f"性能报告已保存至 {report_path}")
        print(f"\n=== 最终性能报告 ===")
        print(f"数据集: {self.dataset_root}")
        print(f"类别数: {self.num_classes}")
        print("\n平均值 (Mean):")
        for metric, value in mean_metrics.items():
            print(f"  {metric}: {value:.4f}")
        print("\n标准差 (Std):")
        for metric, value in std_metrics.items():
            print(f"  {metric}: {value:.4f}")
    
    def _generate_visualizations(self, metrics_df: pd.DataFrame, mean_metrics: Dict, std_metrics: Dict):
        """生成可视化结果"""
        # 1. 指标对比图
        metrics = list(mean_metrics.keys())
        means = list(mean_metrics.values())
        stds = list(std_metrics.values())
        
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(metrics))
        width = 0.35
        
        ax.bar(x - width/2, means, width, label='Mean', yerr=stds, capsize=5)
        ax.set_ylabel('Value')
        ax.set_title('Performance Metrics')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics, rotation=45)
        ax.legend()
        
        plt.tight_layout()
        metrics_plot_path = os.path.join(self.output_dir, "metrics_comparison.png")
        plt.savefig(metrics_plot_path)
        plt.close()
        
        # 2. 各run的性能对比
        run_ids = [f"run_{i}" for i in range(10)]
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        for i, metric in enumerate(metrics):
            metric_values = metrics_df[metric].values
            ax.plot(run_ids, metric_values, marker='o', label=metric)
        
        ax.set_ylabel('Value')
        ax.set_title('Performance Across Runs')
        ax.set_xticks(range(len(run_ids)))
        ax.set_xticklabels(run_ids, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        runs_plot_path = os.path.join(self.output_dir, "runs_performance.png")
        plt.savefig(runs_plot_path)
        plt.close()
        
        print(f"可视化结果已保存至 {metrics_plot_path} 和 {runs_plot_path}")
    
    def run(self):
        """运行完整流程"""
        # 执行网格搜索
        best_params = self.grid_search()
        
        # 使用最优超参数评估所有run_i
        self.evaluate_best_params(best_params)
        
        # 生成报告
        self.generate_report()


def main():
    """主函数"""
    args = parse_args()
    
    # 创建自动化评测实例
    automation = LDLAutomation(args)
    
    # 运行完整流程
    automation.run()


if __name__ == "__main__":
    main()
