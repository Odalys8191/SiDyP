import numpy as np
import torch
from torch import nn
from tqdm import trange
from transformers  import BertForSequenceClassification, get_linear_schedule_with_warmup
from torch.cuda.amp import GradScaler
import torch.nn.functional as F

from utils import euclidean_dist, kl_div


class FeatureModel(nn.Module):
    """
    基于特征的模型类，用于LDL任务
    直接接受feature向量作为输入，输出分类logits和embeddings
    支持多模型分支结构
    """
    def __init__(self, args, feature_dim):
        super().__init__()
        self.args = args
        self.feature_dim = feature_dim
        # 映射特征到BERT隐藏层维度（768）
        self.fc1 = nn.Linear(feature_dim, 768)
        # 分类层
        self.fc2 = nn.Linear(768, args.num_classes)
        
    def forward(self, features, labels=None):
        # 特征映射到BERT隐藏层维度
        embeddings = F.relu(self.fc1(features))
        # 分类预测
        logits = self.fc2(embeddings)
        
        # 模拟BERT模型的输出结构，确保兼容性
        # BERT模型输出：(loss, logits, hidden_states, attentions)
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
            return (loss, logits, (None, None, None, embeddings.unsqueeze(0)))  # 保持输出结构一致
        else:
            return (None, logits, (None, None, None, embeddings.unsqueeze(0)))  # 保持输出结构一致


class NLLModel(nn.Module):
    def __init__(self, args, feature_dim=None):
        """
        初始化NLLModel
        
        Args:
            args: 配置参数
            feature_dim: 特征维度，如果提供则使用FeatureModel，否则使用BERT模型
        """
        super().__init__()
        num_labels = args.num_classes
        self.args = args
        self.models = nn.ModuleList()
        self.loss_fnt = nn.CrossEntropyLoss()
        self.use_feature = feature_dim is not None  # 是否使用feature输入
        
        for _ in range(args.num_model):
            if self.use_feature:
                # 使用基于Feature的模型
                model = FeatureModel(args, feature_dim)
            else:
                # 使用原始BERT模型
                model = BertForSequenceClassification.from_pretrained(args.plc, num_labels=num_labels, output_hidden_states=True)
            model.to(self.args.device)
            self.models.append(model)

    def forward(self, input_ids, attention_mask=None, features=None, labels=None):
        """
        前向传播方法，支持feature输入
        
        Args:
            input_ids: 文本输入的input_ids（仅BERT模型使用）
            attention_mask: 文本输入的attention_mask（仅BERT模型使用）
            features: 特征输入（仅FeatureModel使用）
            labels: 标签
        
        Returns:
            模型输出
        """
        num_models = len(self.models)
        outputs = []
        
        for i in range(num_models):
            if self.use_feature:
                # 使用FeatureModel，输入features
                output = self.models[i](
                    features=features.to(self.args.device),
                    labels=labels.to(self.args.device) if labels is not None else None
                )
            else:
                # 使用BERT模型，输入input_ids和attention_mask
                output = self.models[i](
                    input_ids=input_ids.to(self.args.device),
                    attention_mask=attention_mask.to(self.args.device),
                    labels=labels.to(self.args.device) if labels is not None else None,
                    return_dict=False,
                )
            outputs.append(output)

        model_output = outputs
        if labels is not None:
            loss = sum([output[0] for output in outputs]) / num_models
            logits = [output[1] for output in outputs]
            probs = [F.softmax(logit, dim=-1) for logit in logits]
            avg_prob = torch.stack(probs, dim=0).mean(0)
            reg_loss = sum([kl_div(avg_prob, prob) for prob in probs]) / num_models
            loss = loss + self.args.alpha_t * reg_loss.mean()
            return loss
        return model_output

def accurate_nb(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat)

class PLC_Trainer(nn.Module):
    def __init__(self, args, train_dataloader, valid_dataloader, test_dataloader, feature_dim=None):
        """
        初始化PLC_Trainer
        
        Args:
            args: 配置参数
            train_dataloader: 训练数据加载器
            valid_dataloader: 验证数据加载器
            test_dataloader: 测试数据加载器
            feature_dim: 特征维度，如果提供则使用FeatureModel，否则使用BERT模型
        """
        super().__init__()
        self.args = args
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.test_dataloader = test_dataloader
        self.feature_dim = feature_dim
        self.use_feature = feature_dim is not None  # 是否使用feature输入
        self.model = NLLModel(self.args, feature_dim)
        self.model.to(self.args.device)

    
    def train(self):
        t_total = len(self.train_dataloader) * self.args.plc_epochs
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.plc_lr, eps=1e-9)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.06*t_total), num_training_steps=t_total)
        scaler = GradScaler()

        best_val = -np.inf
        dists_epochs = []
        train_embeds_list = [None] * len(self.model.models)
        eval_embeds_list = [None] * len(self.model.models)
        test_embeds_list = [None] * len(self.model.models)
        dists_list = [None] * len(self.model.models)
        num_epochs = 0

        for epoch in trange(self.args.plc_epochs, desc="Epoch"): 
        # Training
            # Set our model to training mode (as opposed to evaluation mode)
            # Tracking variables
            tr_loss =  0
            nb_tr_examples, nb_tr_steps = 0, 0
            self.model.train()
            # Train the data for one epoch
            for step, batch in enumerate(self.train_dataloader):
                # Add batch to GPU
                batch = tuple(t.to(self.args.device) for t in batch)
                
                # 根据是否使用feature，解包不同的batch数据
                if self.use_feature:
                    # 使用feature输入，batch结构：(features, true_labels, noisy_labels)
                    b_features, b_true_labels, b_noisy_labels = batch
                    # 在LDL模式下，true_labels和noisy_labels是分布，我们取最大概率的类别作为训练标签
                    b_labels = torch.argmax(b_noisy_labels, dim=1) if b_noisy_labels.dim() > 1 else b_noisy_labels
                else:
                    # 使用文本输入，batch结构：(input_ids, attention_mask, true_labels, noisy_labels)
                    b_input_ids, b_input_mask, _, b_labels = batch

                if num_epochs < int(self.args.plc_epochs/10):
                    self.args.alpha_t = 0
                else:
                    self.args.alpha_t = self.args.alpha_t
                
                # 根据是否使用feature，调用model的不同参数
                if self.use_feature:
                    loss_ce = self.model(input_ids=None, attention_mask=None, features=b_features, labels=b_labels)
                else:
                    loss_ce = self.model(b_input_ids, attention_mask=b_input_mask, labels=b_labels)
                if torch.cuda.device_count() > 1:
                    loss_ce = loss_ce.mean()
                scaler.scale(loss_ce).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                self.model.zero_grad()
                tr_loss += loss_ce.item()
                
                # 根据输入类型更新样本数
                if self.use_feature:
                    nb_tr_examples += b_features.size(0)
                else:
                    nb_tr_examples += b_input_ids.size(0)
                    
                nb_tr_steps += 1
            print("Train cross entropy loss: {}".format(tr_loss/nb_tr_steps))
            num_epochs += 1

            # Validation
            # Put model in evaluation mode to evaluate loss on the validation set
            self.model.eval()
            train_embeds = [None] * len(self.model.models)
            train_labels = [None] * len(self.model.models)
            train_logits = [None] * len(self.model.models)
            for batch in self.train_dataloader:
                batch = tuple(t.to(self.args.device) for t in batch)
                
                # 根据是否使用feature，解包不同的batch数据
                if self.use_feature:
                    b_features, b_true_labels, b_noisy_labels = batch
                    b_labels = torch.argmax(b_noisy_labels, dim=1) if b_noisy_labels.dim() > 1 else b_noisy_labels
                else:
                    b_input_ids, b_input_mask, _, b_labels = batch
                
                with torch.no_grad():
                    # 根据是否使用feature，调用model的不同参数
                    if self.use_feature:
                        outputs = self.model(input_ids=None, attention_mask=None, features=b_features)
                    else:
                        outputs = self.model(b_input_ids, attention_mask=b_input_mask)
                    
                    for idx in range(len(outputs)):
                        # 在FeatureModel中，logits是output[1]，而在BERT模型中是output[0]
                        logits = outputs[idx][1] if self.use_feature else outputs[idx][0]
                        if train_embeds[idx] == None:
                            # 提取embeddings：在BERT中是output[-1][-1][:,0,:]，在FeatureModel中是output[-1][-1]
                            embeddings = outputs[idx][-1][-1][:,0,:].squeeze() if not self.use_feature else outputs[idx][-1][-1].squeeze()
                            train_embeds[idx] = embeddings
                            train_labels[idx] = b_labels
                            train_logits[idx] = F.softmax(logits, dim=-1)
                        else:
                            embeddings = outputs[idx][-1][-1][:,0,:].squeeze() if not self.use_feature else outputs[idx][-1][-1].squeeze()
                            train_embeds[idx] = torch.cat((train_embeds[idx], embeddings), 0)
                            train_labels[idx] = torch.cat((train_labels[idx], b_labels), 0)
                            train_logits[idx] = torch.cat((train_logits[idx], F.softmax(logits, dim=-1)), 0)
            for idx in range(len(outputs)):
                if train_embeds_list[idx] == None:
                    train_embeds_list[idx] = torch.zeros((self.args.plc_epochs, train_embeds[idx].shape[0], train_embeds[idx].shape[1]))
                    train_embeds_list[idx][0] = train_embeds[idx].detach()
                else:
                    train_embeds_list[idx][epoch] = train_embeds[idx].detach()
            # Tracking variables 
            eval_accurate_nb = 0
            nb_eval_examples = 0
            logits_list = []
            labels_list = []

            self.model.eval()
            eval_embeds = [None] * len(self.model.models)
            eval_labels = [None] * len(self.model.models)
            eval_logits = [None] * len(self.model.models)
            # Evaluate data for one epoch
            for batch in self.valid_dataloader:
                # Add batch to GPU
                batch = tuple(t.to(self.args.device) for t in batch)
                
                # 根据是否使用feature，解包不同的batch数据
                if self.use_feature:
                    # 使用feature输入，batch结构：(features, true_labels, noisy_labels)
                    b_features, b_true_labels, b_noisy_labels = batch
                    # 在LDL模式下，true_labels和noisy_labels是分布，我们取最大概率的类别作为训练标签
                    b_labels = torch.argmax(b_noisy_labels, dim=1) if b_noisy_labels.dim() > 1 else b_noisy_labels
                else:
                    # 使用文本输入，batch结构：(input_ids, attention_mask, true_labels, noisy_labels)
                    b_input_ids, b_input_mask, _, b_labels = batch
                
                # Telling the model not to compute or store gradients, saving memory and speeding up validation
                with torch.no_grad():
                    # 根据是否使用feature，调用model的不同参数
                    if self.use_feature:
                        outputs = self.model(input_ids=None, attention_mask=None, features=b_features)
                    else:
                        outputs = self.model(b_input_ids, attention_mask=b_input_mask)
                    
                    for idx in range(len(outputs)):
                        # 在FeatureModel中，logits是output[1]，而在BERT模型中是output[0]
                        logits = outputs[idx][1] if self.use_feature else outputs[idx][0]
                        if eval_embeds[idx] == None:
                            # 提取embeddings：在BERT中是output[-1][-1][:,0,:]，在FeatureModel中是output[-1][-1]
                            embeddings = outputs[idx][-1][-1][:,0,:].squeeze() if not self.use_feature else outputs[idx][-1][-1].squeeze()
                            eval_embeds[idx] = embeddings
                            eval_labels[idx] = b_labels
                            eval_logits[idx] = F.softmax(logits, dim=-1)
                        else:
                            embeddings = outputs[idx][-1][-1][:,0,:].squeeze() if not self.use_feature else outputs[idx][-1][-1].squeeze()
                            eval_embeds[idx] = torch.cat((eval_embeds[idx], embeddings), 0)
                            eval_labels[idx] = torch.cat((eval_labels[idx], b_labels), 0)
                            eval_logits[idx] = torch.cat((eval_logits[idx], F.softmax(logits, dim=-1)), 0)
                    
                    # 获取最后一个模型的logits用于评估
                    logits = [output[1] if self.use_feature else output[0] for output in outputs]
                    logits = logits[-1]
                    logits_list.append(logits)
                    labels_list.append(b_labels)
                # Move logits and labels to CPU
                logits = logits.detach().cpu().numpy()
                label_ids = b_labels.to('cpu').numpy()

                tmp_eval_nb = accurate_nb(logits, label_ids)
        
                eval_accurate_nb += tmp_eval_nb
                nb_eval_examples += label_ids.shape[0]
            for idx in range(len(self.model.models)):
                if eval_embeds_list[idx] == None:
                    eval_embeds_list[idx] = torch.zeros((self.args.plc_epochs, eval_embeds[idx].shape[0], eval_embeds[idx].shape[1]))
                    eval_embeds_list[idx][0] = eval_embeds[idx].detach()
                else:
                    eval_embeds_list[idx][epoch] = eval_embeds[idx].detach()
            eval_accuracy = eval_accurate_nb/nb_eval_examples
            print("Validation Accuracy: {}".format(eval_accuracy))
            scheduler.step(eval_accuracy)

            if eval_accuracy > best_val:
                best_val = eval_accuracy
                best_model = self.model
                

            # Put model in evaluation mode
            self.model.eval()
            # Tracking variables 
            eval_accurate_nb = 0
            nb_test_examples = 0
            logits_list = []
            labels_list = []
            test_embeds = [None] * len(self.model.models)
            test_labels = [None] * len(self.model.models)
            # Predict 
            for batch in self.test_dataloader:
                # Add batch to GPU
                batch = tuple(t.to(self.args.device) for t in batch)
                
                # 根据是否使用feature，解包不同的batch数据
                if self.use_feature:
                    # 使用feature输入，batch结构：(features, true_labels)
                    b_features, b_true_labels = batch
                    # 在LDL模式下，true_labels是分布，我们取最大概率的类别作为测试标签
                    b_labels = torch.argmax(b_true_labels, dim=1) if b_true_labels.dim() > 1 else b_true_labels
                else:
                    # 使用文本输入，batch结构：(input_ids, input_mask, true_labels)
                    b_input_ids, b_input_mask, b_labels = batch
                
                # Telling the model not to compute or store gradients, saving memory and speeding up prediction
                with torch.no_grad():
                    # 根据是否使用feature，调用model的不同参数
                    if self.use_feature:
                        outputs = self.model(input_ids=None, attention_mask=None, features=b_features)
                    else:
                        outputs = self.model(b_input_ids, attention_mask=b_input_mask)
                    
                    # 获取logits：在FeatureModel中是output[1]，在BERT模型中是output[0]
                    logits = [output[1] if self.use_feature else output[0] for output in outputs]
                    logits = logits[-1] #torch.stack(logits, dim=0).mean(0)
                    
                    for idx in range(len(outputs)):
                        if test_embeds[idx] == None:
                            # 提取embeddings：在BERT中是output[-1][-1][:,0,:]，在FeatureModel中是output[-1][-1]
                            embeddings = outputs[idx][-1][-1][:,0,:].squeeze() if not self.use_feature else outputs[idx][-1][-1].squeeze()
                            test_embeds[idx] = embeddings
                            test_labels[idx] = b_labels
                        else:
                            embeddings = outputs[idx][-1][-1][:,0,:].squeeze() if not self.use_feature else outputs[idx][-1][-1].squeeze()
                            test_embeds[idx] = torch.cat((test_embeds[idx], embeddings), 0)
                            test_labels[idx] = torch.cat((test_labels[idx], b_labels), 0)
                    logits_list.append(logits)
                    labels_list.append(b_labels)
                # Move logits and labels to CPU
                logits = logits.detach().cpu().numpy()
                label_ids = b_labels.to('cpu').numpy()

                tmp_eval_nb = accurate_nb(logits, label_ids)
                eval_accurate_nb += tmp_eval_nb
                nb_test_examples += label_ids.shape[0]
            for idx in range(len(outputs)):
                if test_embeds_list[idx] == None:
                    test_embeds_list[idx] = torch.zeros((self.args.plc_epochs, test_embeds[idx].shape[0], test_embeds[idx].shape[1]))
                    test_embeds_list[idx][0] = test_embeds[idx].detach()
                else:
                    test_embeds_list[idx][epoch] = test_embeds[idx].detach()

            print("Test Accuracy: {}".format(eval_accurate_nb/nb_test_examples))

            full_dists = [None] * len(self.model.models)
            for idx in range(len(self.model.models)):
                dists_embeds = torch.cat((train_embeds[idx], eval_embeds[idx], test_embeds[idx]), 0)
                dists_labels = torch.cat((train_labels[idx], eval_labels[idx], test_labels[idx]), 0)
                dists = euclidean_dist(self.args, dists_embeds, dists_labels)
                full_dists[idx] = dists
                dists = [dists[i][dists_labels[i]] for i in range(len(dists))]
                dists_epochs.append(dists)
            
                if dists_list[idx] is None:
                    dists_list[idx] = torch.zeros((self.args.plc_epochs, full_dists[idx].shape[0], full_dists[idx].shape[1]))
                    dists_list[idx][0] = full_dists[idx].detach()
                else:
                    dists_list[idx][epoch] = full_dists[idx].detach()

        train_embeds_list = torch.stack(train_embeds_list, dim=0)
        eval_embeds_list = torch.stack(eval_embeds_list, dim=0)
        test_embeds_list = torch.stack(test_embeds_list, dim=0)
        dists_list = torch.stack(dists_list, dim=0)
        return train_embeds_list, eval_embeds_list, test_embeds_list, best_model, dists_list
