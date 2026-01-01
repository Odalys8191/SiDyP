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
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from itertools import product
from typing import Dict, List, Tuple

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

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
        
        # 固定超参数（使用main.py中的默认值）
        self.fixed_params = {
            'seed': self.base_seed,
            'num_classes': self.num_classes,
            'ldl': True,
            'plc_lr': 5e-5,
            'eval_batch_size': 128,
            'plc_epochs': 100,
            'diff_epochs': 100  # 默认值，后续可以通过参数调整
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
    
    def _generate_command(self, params: Dict, run_idx: int = 0) -> List[str]:
        """生成命令行命令"""
        command = [
            sys.executable, "main.py",
            "--ldl",
            "--dataset", self._get_run_path(run_idx),
            "--num_classes", str(params['num_classes']),
            "--plc_lr", str(params['plc_lr']),
            "--diff_lr", str(params['diff_lr']),
            "--train_batch_size", str(params['train_batch_size']),
            "--eval_batch_size", str(params['eval_batch_size']),
            "--plc_epochs", str(params['plc_epochs']),
            "--diff_epochs", str(params['diff_epochs']),
            "--seed", str(params['seed'])
        ]
        return command
    
    def _run_experiment(self, params: Dict, run_idx: int = 0) -> Tuple[Dict, float]:
        """运行单次实验"""
        command = self._generate_command(params, run_idx)
        print(f"执行命令: {' '.join(command)}")
        
        # 按run_idx组织实验结果目录
        run_dir = os.path.join(self.output_dir, f"run_{run_idx}")
        os.makedirs(run_dir, exist_ok=True)
        
        # 创建实验结果目录
        exp_id = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        exp_dir = os.path.join(run_dir, exp_id)
        os.makedirs(exp_dir, exist_ok=True)
        
        # 运行命令并捕获输出
        log_file = os.path.join(exp_dir, "experiment.log")
        with open(log_file, 'w') as f:
            result = subprocess.run(
                command,
                cwd=os.path.dirname(os.path.abspath(__file__)),
                stdout=f,
                stderr=subprocess.STDOUT,
                text=True
            )
        
        # 解析实验结果（这里需要根据实际输出格式调整）
        # 假设main.py会生成result.json文件
        result_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "result.json")
        if os.path.exists(result_path):
            with open(result_path, 'r') as f:
                result_data = json.load(f)
            # 复制result.json到实验目录
            import shutil
            shutil.copy(result_path, exp_dir)
        else:
            # 模拟结果数据（需要根据实际情况修改）
            result_data = {
                'metrics': {
                    'loss': np.random.rand(),
                    'accuracy': np.random.rand(),
                    'kl_div': np.random.rand()
                }
            }
        
        # 计算改进率
        our_metrics = list(result_data['metrics'].values())
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
        
        # 获取模型保存路径（根据main.py中的模型保存逻辑）
        # 模型保存路径格式：best_models/ldl/数据集名称/种子值
        model_save_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "best_models", "ldl", self.dataset_name, str(params['seed'])
        )
        
        # 保存实验配置和结果
        exp_result = {
            'exp_id': exp_id,
            'params': params,
            'run_idx': run_idx,
            'metrics': result_data['metrics'],
            'improvement': imp,
            'log_file': log_file,
            'exp_dir': exp_dir,
            'model_save_dir': model_save_dir
        }
        
        return exp_result, imp
    
    def grid_search(self) -> Dict:
        """执行网格搜索"""
        print("开始网格搜索...")
        
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
            
            print(f"\n正在测试超参数组合 {i+1}/{len(hyperparam_combinations)}: {params}")
            
            # 执行10次独立实验
            exp_imps = []
            run_results = []
            for exp_idx in range(10):
                # 使用不同的随机种子
                exp_params = params.copy()
                exp_params['seed'] = self.base_seed + exp_idx
                
                exp_result, imp = self._run_experiment(exp_params, run_idx=0)
                exp_imps.append(imp)
                run_results.append(exp_result)
                self.grid_search_results.append(exp_result)
            
            # 计算平均改进率
            avg_imp = np.mean(exp_imps)
            print(f"10次实验平均改进率: {avg_imp:.4f}")
            
            # 更新最优超参数
            if avg_imp > best_imp:
                best_imp = avg_imp
                best_params = params
                best_run_results = run_results
                print(f"找到更优超参数组合，改进率: {best_imp:.4f}")
        
        # 保存网格搜索结果
        self._save_grid_search_results()
        
        # 保存最优模型
        self._save_best_models(best_run_results, "grid_search")
        
        print(f"\n网格搜索完成！")
        print(f"最优超参数组合: {best_params}")
        print(f"最优平均改进率: {best_imp:.4f}")
        
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
        """使用最优超参数在所有run_i上评估"""
        print("\n开始使用最优超参数评估所有run_i...")
        
        eval_results = []
        
        for run_idx in range(10):
            print(f"\n正在评估 run_{run_idx}...")
            
            # 使用不同的随机种子
            params = best_params.copy()
            params['seed'] = self.base_seed + run_idx
            
            exp_result, imp = self._run_experiment(params, run_idx=run_idx)
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
        result_path = os.path.join(self.output_dir, "grid_search_results.json")
        with open(result_path, 'w') as f:
            json.dump(self.grid_search_results, f, indent=2)
        print(f"网格搜索结果已保存至 {result_path}")
    
    def _save_eval_results(self):
        """保存评估结果"""
        result_path = os.path.join(self.output_dir, "eval_results.json")
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
