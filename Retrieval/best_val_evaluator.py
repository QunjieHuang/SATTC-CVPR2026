"""
Best-Val评估协议实现
用于ICASSP论文的严格评估标准
"""

import os
import json
import torch
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
import itertools
import datetime
from eegdatasets_leaveone import EEGDataset

class BestValEEGDataset(Dataset):
    """
    支持Best-Val的EEG数据集
    """
    def __init__(self, data_list, base_dataset):
        self.data_list = data_list
        self.text_features = base_dataset.text_features
        self.img_features = base_dataset.img_features
        self.img_proto = base_dataset.img_proto
        self.text_proto = base_dataset.text_proto
        
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        return self.data_list[idx][:6]  # 排除subject信息


def create_best_val_splits(data_path, exclude_subject, all_subjects, val_ratio=0.15, seed=42):
    """
    创建LOSO + 内层验证的数据划分
    Args:
        data_path: 数据路径
        exclude_subject: 测试被试
        all_subjects: 所有被试列表
        val_ratio: 验证集比例
        seed: 随机种子
    Returns:
        train_data, val_data: 训练和验证数据
    """
    # 训练被试（除了测试被试）
    train_subjects = [s for s in all_subjects if s != exclude_subject]
    
    # 为每个训练被试收集所有trial
    all_train_data = []
    all_train_labels = []
    
    print(f"📊 Creating Best-Val splits for test subject: {exclude_subject}")
    print(f"   Training subjects: {train_subjects}")
    
    for subject in train_subjects:
        # 加载该被试的所有数据
        subject_dataset = EEGDataset(data_path, subjects=[subject], train=True)
        subject_count = 0
        
        for i in range(len(subject_dataset)):
            eeg_data, label, text, text_feat, img, img_feat = subject_dataset[i]
            all_train_data.append((eeg_data, label, text, text_feat, img, img_feat, subject))
            all_train_labels.append(label)
            subject_count += 1
        
        print(f"   {subject}: {subject_count} trials")
    
    # 设置随机种子确保可重复
    np.random.seed(seed)
    indices = np.arange(len(all_train_data))
    np.random.shuffle(indices)
    
    # 按比例划分
    val_size = int(len(all_train_data) * val_ratio)
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]
    
    # 创建训练和验证数据
    train_data = [all_train_data[i] for i in train_indices]
    val_data = [all_train_data[i] for i in val_indices]
    
    print(f"   📊 Final split:")
    print(f"      Training: {len(train_data)} trials ({(1-val_ratio)*100:.1f}%)")
    print(f"      Validation: {len(val_data)} trials ({val_ratio*100:.1f}%)")
    
    return train_data, val_data


def best_val_train_loop(sub, current_time, eeg_model, train_dataloader, val_dataloader, test_dataloader, 
                       optimizer, device, text_features_train_all, text_features_test_all, 
                       img_features_train_all, img_features_test_all, config, scheduler=None):
    """
    Best-Val训练循环：基于验证集性能选择最佳模型
    """
    from Retrieval.run_sattc_loso import train_model, evaluate_model
    
    # 最佳模型跟踪
    best_val_top5 = 0.0
    best_val_epoch = 0
    best_model_state = None
    
    # 存储路径
    if config.insubject:
        checkpoint_dir = f"./checkpoints/best_val/{config.encoder_type}/{sub}/{current_time}"
    else:
        checkpoint_dir = f"./checkpoints/best_val/across/{config.encoder_type}/{current_time}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    print(f"📁 Checkpoints will be saved to: {checkpoint_dir}")
    
    for epoch in range(config.epochs):
        config._current_epoch = epoch
        current_lr = optimizer.param_groups[0]['lr']
        
        # 训练阶段
        train_loss, train_accuracy, features_tensor = train_model(
            sub, eeg_model, train_dataloader, optimizer, device, 
            text_features_train_all, img_features_train_all, config=config
        )
        
        # 验证阶段 - 使用Top-5作为选择标准
        val_loss, val_top1, val_top5 = evaluate_model(
            sub, eeg_model, val_dataloader, device, 
            text_features_train_all, img_features_train_all, k=200, config=config
        )
        
        # 🔥 Best-Val逻辑：基于验证集Top-5选择最佳模型
        if val_top5 > best_val_top5:
            best_val_top5 = val_top5
            best_val_epoch = epoch + 1
            
            # 保存最佳模型状态
            best_model_state = {
                'epoch': epoch + 1,
                'model_state_dict': eeg_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'best_val_top5': best_val_top5,
                'val_top1': val_top1,
                'train_accuracy': train_accuracy,
                'config_dict': vars(config),
                'random_state': {
                    'torch': torch.get_rng_state(),
                    'numpy': np.random.get_state(),
                    'python': random.getstate()
                }
            }
            
            # 保存checkpoint
            best_checkpoint_path = f"{checkpoint_dir}/best_model_{sub}_seed{config.seed}.pth"
            torch.save(best_model_state, best_checkpoint_path)
            
            print(f"🎯 New best validation Top-5: {val_top5:.4f} (epoch {epoch+1})")
        
        # 学习率调度
        if scheduler is not None:
            if hasattr(scheduler, 'step'):
                if hasattr(scheduler, 'current_epoch'):
                    new_lr, phase = scheduler.step()
                else:
                    scheduler.step()
                    new_lr = optimizer.param_groups[0]['lr']
        
        # 简化的输出
        print(f"Epoch {epoch + 1:2d}/{config.epochs} - Train Acc: {train_accuracy:.4f} | Val Top-1: {val_top1:.4f}, Top-5: {val_top5:.4f} | Best: {best_val_top5:.4f} (ep{best_val_epoch})")
    
    # 🔥 加载最佳模型进行最终测试
    print(f"\n🏆 Loading best model from epoch {best_val_epoch} (Val Top-5: {best_val_top5:.4f})")
    eeg_model.load_state_dict(best_model_state['model_state_dict'])
    
    # 最终测试评估
    test_loss, test_top1, test_top5 = evaluate_model(
        sub, eeg_model, test_dataloader, device, 
        text_features_test_all, img_features_test_all, k=200, config=config
    )
    
    # k-way评估
    _, v2_acc, _ = evaluate_model(sub, eeg_model, test_dataloader, device, text_features_test_all, img_features_test_all, k=2, config=config)
    _, v4_acc, _ = evaluate_model(sub, eeg_model, test_dataloader, device, text_features_test_all, img_features_test_all, k=4, config=config)
    _, v10_acc, _ = evaluate_model(sub, eeg_model, test_dataloader, device, text_features_test_all, img_features_test_all, k=10, config=config)
    
    final_results = {
        'subject': sub,
        'seed': config.seed,
        'best_val_epoch': best_val_epoch,
        'best_val_top5': best_val_top5,
        'test_top1': test_top1,
        'test_top5': test_top5,
        'test_v2': v2_acc,
        'test_v4': v4_acc,
        'test_v10': v10_acc,
        'checkpoint_path': best_checkpoint_path,
        'experiment_time': current_time
    }
    
    print(f"\n📊 Final Test Results (Best-Val from epoch {best_val_epoch}):")
    print(f"   Test Top-1: {test_top1:.4f}")
    print(f"   Test Top-5: {test_top5:.4f}")
    print(f"   Test v2: {v2_acc:.4f}, v4: {v4_acc:.4f}, v10: {v10_acc:.4f}")
    
    return final_results


def run_best_val_single_subject(sub, args, current_time, device):
    """
    对单个被试运行Best-Val评估
    """
    from Retrieval.run_sattc_loso import SATTC, WarmupCosineAnnealingLR, set_seed
    
    print(f"\n📍 Best-Val LOSO for subject: {sub} (seed: {args.seed})")
    print(f"{'='*70}")
    
    try:
        # 设置随机种子
        set_seed(args.seed)
        
        # 🔥 创建Best-Val数据划分
        base_train_dataset = EEGDataset(args.data_path, exclude_subject=sub, subjects=args.subjects, train=True)
        test_dataset = EEGDataset(args.data_path, exclude_subject=sub, subjects=args.subjects, train=False)
        
        train_data, val_data = create_best_val_splits(
            args.data_path, exclude_subject=sub, all_subjects=args.subjects, 
            val_ratio=0.15, seed=args.seed
        )
        
        # 创建数据集和加载器
        train_dataset = BestValEEGDataset(train_data, base_train_dataset)
        val_dataset = BestValEEGDataset(val_data, base_train_dataset)
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=True)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=True)
        
        # 模型和优化器初始化
        eeg_model = SATTC()
        eeg_model.to(device)
        
        initial_lr = args.warmup_start_lr if args.use_warmup else args.lr
        optimizer = AdamW(itertools.chain(eeg_model.parameters()), lr=initial_lr)
        
        scheduler = None
        if args.use_warmup and args.use_cosine_annealing:
            scheduler = WarmupCosineAnnealingLR(
                optimizer=optimizer,
                warmup_epochs=args.warmup_epochs,
                max_epochs=args.epochs,
                target_lr=args.lr,
                warmup_start_lr=args.warmup_start_lr,
                eta_min=args.eta_min
            )
        
        # 🔥 Best-Val训练
        result = best_val_train_loop(
            sub, current_time, eeg_model, train_loader, val_loader, test_loader,
            optimizer, device, 
            train_dataset.text_features, test_dataset.text_features,
            train_dataset.img_features, test_dataset.img_features,
            config=args, scheduler=scheduler
        )
        
        return result
        
    except Exception as e:
        print(f"❌ Error for {sub}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def save_best_val_results(results, args, current_time):
    """
    保存Best-Val结果到JSON文件
    """
    # 创建结果目录
    results_dir = f"./results/best_val"
    os.makedirs(results_dir, exist_ok=True)
    
    # 构建文件名
    method_name = "SATTC_baseline" if not getattr(args, 'use_arcface', False) else "SATTC_arcface"
    filename = f"{method_name}_seed{args.seed}_{current_time}.json"
    filepath = os.path.join(results_dir, filename)
    
    # 转换为可序列化格式
    serializable_results = []
    for result in results:
        if result is not None:
            # 移除不可序列化的部分
            clean_result = {k: v for k, v in result.items() if k != 'random_state'}
            # 转换numpy类型
            for key, value in clean_result.items():
                if isinstance(value, (np.float32, np.float64, np.int64)):
                    clean_result[key] = float(value) if 'float' in str(type(value)) else int(value)
            
            serializable_results.append(clean_result)
    
    # 添加实验元信息
    experiment_info = {
        'experiment_time': current_time,
        'seed': args.seed,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'use_arcface': getattr(args, 'use_arcface', False),
        'subjects_tested': [r['subject'] for r in serializable_results if r],
        'total_subjects': len(serializable_results),
        'method': method_name
    }
    
    final_data = {
        'experiment_info': experiment_info,
        'results': serializable_results
    }
    
    # 保存到文件
    with open(filepath, 'w') as f:
        json.dump(final_data, f, indent=2)
    
    print(f"💾 Best-Val results saved to: {filepath}")
    return filepath
