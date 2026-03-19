"""
Best-Val evaluation protocol.
Strict leave-one-subject-out evaluation with inner validation split.
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
    EEG dataset wrapper for Best-Val evaluation.
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
        return self.data_list[idx][:6]  # Exclude subject metadata.


def create_best_val_splits(data_path, exclude_subject, all_subjects, val_ratio=0.15, seed=42):
    """
    Create LOSO + inner-validation data splits.
    Args:
        data_path: path to the preprocessed EEG data.
        exclude_subject: subject held out for testing.
        all_subjects: list of all subject identifiers.
        val_ratio: fraction of training data used for validation.
        seed: random seed for reproducibility.
    Returns:
        train_data, val_data: training and validation data lists.
    """
    # Training subjects (all except the held-out test subject).
    train_subjects = [s for s in all_subjects if s != exclude_subject]
    
    # Collect all trials for each training subject.
    all_train_data = []
    all_train_labels = []
    
    print(f" Creating Best-Val splits for test subject: {exclude_subject}")
    print(f"   Training subjects: {train_subjects}")
    
    for subject in train_subjects:
        # Load all trials for this subject.
        subject_dataset = EEGDataset(data_path, subjects=[subject], train=True)
        subject_count = 0
        
        for i in range(len(subject_dataset)):
            eeg_data, label, text, text_feat, img, img_feat = subject_dataset[i]
            all_train_data.append((eeg_data, label, text, text_feat, img, img_feat, subject))
            all_train_labels.append(label)
            subject_count += 1
        
        print(f"   {subject}: {subject_count} trials")
    
    # Set random seed for reproducibility.
    np.random.seed(seed)
    indices = np.arange(len(all_train_data))
    np.random.shuffle(indices)
    
    # Split into train / val by ratio.
    val_size = int(len(all_train_data) * val_ratio)
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]
    
    # Build train and val data lists.
    train_data = [all_train_data[i] for i in train_indices]
    val_data = [all_train_data[i] for i in val_indices]
    
    print(f"    Final split:")
    print(f"      Training: {len(train_data)} trials ({(1-val_ratio)*100:.1f}%)")
    print(f"      Validation: {len(val_data)} trials ({val_ratio*100:.1f}%)")
    
    return train_data, val_data


def best_val_train_loop(sub, current_time, eeg_model, train_dataloader, val_dataloader, test_dataloader, 
                       optimizer, device, text_features_train_all, text_features_test_all, 
                       img_features_train_all, img_features_test_all, config, scheduler=None):
    """
    Best-Val training loop: selects the best checkpoint based on validation performance.
    """
    from Retrieval.run_sattc_loso import train_model, evaluate_model
    
    # Best-model tracking.
    best_val_top5 = 0.0
    best_val_epoch = 0
    best_model_state = None
    
    # Checkpoint path.
    if config.insubject:
        checkpoint_dir = f"./checkpoints/best_val/{config.encoder_type}/{sub}/{current_time}"
    else:
        checkpoint_dir = f"./checkpoints/best_val/across/{config.encoder_type}/{current_time}"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    print(f" Checkpoints will be saved to: {checkpoint_dir}")
    
    for epoch in range(config.epochs):
        config._current_epoch = epoch
        current_lr = optimizer.param_groups[0]['lr']
        
        # Training phase.
        train_loss, train_accuracy, features_tensor = train_model(
            sub, eeg_model, train_dataloader, optimizer, device, 
            text_features_train_all, img_features_train_all, config=config
        )
        
        # Validation phase - use Top-5 as the selection criterion.
        val_loss, val_top1, val_top5 = evaluate_model(
            sub, eeg_model, val_dataloader, device, 
            text_features_train_all, img_features_train_all, k=200, config=config
        )
        
        #  Best-ValTop-5
        if val_top5 > best_val_top5:
            best_val_top5 = val_top5
            best_val_epoch = epoch + 1
            
            # Save best model state.
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
            
            # Save checkpoint to disk.
            best_checkpoint_path = f"{checkpoint_dir}/best_model_{sub}_seed{config.seed}.pth"
            torch.save(best_model_state, best_checkpoint_path)
            
            print(f" New best validation Top-5: {val_top5:.4f} (epoch {epoch+1})")
        
        # Learning-rate scheduling.
        if scheduler is not None:
            if hasattr(scheduler, 'step'):
                if hasattr(scheduler, 'current_epoch'):
                    new_lr, phase = scheduler.step()
                else:
                    scheduler.step()
                    new_lr = optimizer.param_groups[0]['lr']
        
        # Compact per-epoch log.
        print(f"Epoch {epoch + 1:2d}/{config.epochs} - Train Acc: {train_accuracy:.4f} | Val Top-1: {val_top1:.4f}, Top-5: {val_top5:.4f} | Best: {best_val_top5:.4f} (ep{best_val_epoch})")
    

    print(f"\n Loading best model from epoch {best_val_epoch} (Val Top-5: {best_val_top5:.4f})")
    eeg_model.load_state_dict(best_model_state['model_state_dict'])
    
    # Final test evaluation.
    test_loss, test_top1, test_top5 = evaluate_model(
        sub, eeg_model, test_dataloader, device, 
        text_features_test_all, img_features_test_all, k=200, config=config
    )
    
    # k-way accuracy evaluation.
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
    
    print(f"\n Final Test Results (Best-Val from epoch {best_val_epoch}):")
    print(f"   Test Top-1: {test_top1:.4f}")
    print(f"   Test Top-5: {test_top5:.4f}")
    print(f"   Test v2: {v2_acc:.4f}, v4: {v4_acc:.4f}, v10: {v10_acc:.4f}")
    
    return final_results


def run_best_val_single_subject(sub, args, current_time, device):
    """
    Run Best-Val evaluation for a single subject.
    """
    from Retrieval.run_sattc_loso import SATTC, WarmupCosineAnnealingLR, set_seed
    
    print(f"\n Best-Val LOSO for subject: {sub} (seed: {args.seed})")
    print(f"{'='*70}")
    
    try:
        # Set random seed.
        set_seed(args.seed)
        
        #  Best-Val
        base_train_dataset = EEGDataset(args.data_path, exclude_subject=sub, subjects=args.subjects, train=True)
        test_dataset = EEGDataset(args.data_path, exclude_subject=sub, subjects=args.subjects, train=False)
        
        train_data, val_data = create_best_val_splits(
            args.data_path, exclude_subject=sub, all_subjects=args.subjects, 
            val_ratio=0.15, seed=args.seed
        )
        
        # Create datasets and data loaders.
        train_dataset = BestValEEGDataset(train_data, base_train_dataset)
        val_dataset = BestValEEGDataset(val_data, base_train_dataset)
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=True)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=True)
        
        # Initialise model and optimizer.
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
        
        #  Best-Val
        result = best_val_train_loop(
            sub, current_time, eeg_model, train_loader, val_loader, test_loader,
            optimizer, device, 
            train_dataset.text_features, test_dataset.text_features,
            train_dataset.img_features, test_dataset.img_features,
            config=args, scheduler=scheduler
        )
        
        return result
        
    except Exception as e:
        print(f" Error for {sub}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def save_best_val_results(results, args, current_time):
    """
    Save Best-Val results to a JSON file.
    """
    # Create results directory.
    results_dir = f"./results/best_val"
    os.makedirs(results_dir, exist_ok=True)
    
    # Build output filename.
    method_name = "SATTC_baseline" if not getattr(args, 'use_arcface', False) else "SATTC_arcface"
    filename = f"{method_name}_seed{args.seed}_{current_time}.json"
    filepath = os.path.join(results_dir, filename)
    
    # Convert results to a JSON-serialisable format.
    serializable_results = []
    for result in results:
        if result is not None:
            # Remove non-serialisable entries.
            clean_result = {k: v for k, v in result.items() if k != 'random_state'}
            # Convert numpy scalar types.
            for key, value in clean_result.items():
                if isinstance(value, (np.float32, np.float64, np.int64)):
                    clean_result[key] = float(value) if 'float' in str(type(value)) else int(value)
            
            serializable_results.append(clean_result)
    
    # Add experiment metadata.
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
    
    # Write to disk.
    with open(filepath, 'w') as f:
        json.dump(final_data, f, indent=2)
    
    print(f" Best-Val results saved to: {filepath}")
    return filepath
