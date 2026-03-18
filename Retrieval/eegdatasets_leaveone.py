import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
# import clip
from torch.nn import functional as F
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import json

cuda_device_count = torch.cuda.device_count()
print(cuda_device_count)
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model_type = 'ViT-H-14'

import json

# Resolve paths relative to this file to avoid cwd issues when running different scripts
_this_dir = os.path.dirname(os.path.abspath(__file__))

# Load the configuration from the JSON file (always relative to this module)
config_path = os.path.join(_this_dir, "data_config.json")
with open(config_path, "r", encoding="utf-8") as config_file:
    config = json.load(config_file)


def _resolve(path_str: str) -> str:
    """Resolve absolute/relative paths against this module directory."""
    if os.path.isabs(path_str):
        return path_str
    return os.path.normpath(os.path.join(_this_dir, path_str))


# Access and resolve the paths from the config
data_path = _resolve(config["data_path"])
img_directory_training = _resolve(config["img_directory_training"])
img_directory_test = _resolve(config["img_directory_test"])


class EEGDataset(Dataset):
    """
    subjects = ['sub-01', 'sub-02', 'sub-05', 'sub-04', 'sub-03', 'sub-06', 'sub-07', 'sub-08', 'sub-09', 'sub-10']
    """
    def __init__(self, data_path, exclude_subject=None, subjects=None, train=True, time_window=[0, 1.0], classes = None, pictures = None, val_size=None, anchor_mode=None, return_subject_ids=False):
        self.data_path = data_path
        self.train = train
        self.subject_list = os.listdir(data_path)
        self.subjects = self.subject_list if subjects is None else subjects
        self.n_sub = len(self.subjects)
        self.time_window = time_window
        self.n_cls = 1654 if train else 200
        self.classes = classes
        self.pictures = pictures
        self.exclude_subject = exclude_subject  
        self.val_size = val_size
        self.anchor_mode = anchor_mode
        if self.anchor_mode not in (None, 'first_per_class'):
            raise ValueError(f"Unsupported anchor_mode: {self.anchor_mode}")
        self.return_subject_ids = return_subject_ids
        self.subject_to_idx = {subj: idx for idx, subj in enumerate(self.subjects)}
        self.idx_to_subject = {idx: subj for subj, idx in self.subject_to_idx.items()}
        self.sample_subject_indices = None
        self.images_per_class = None
        # assert any subjects in subject_list
        assert any(sub in self.subject_list for sub in self.subjects)

        self.data, self.labels, self.text, self.img = self.load_data()
        
        self.data = self.extract_eeg(self.data, time_window)
        
        
        # Load precomputed features from local files (always required in local-only mode)
        features_filename = f'{model_type}_features_train.pt' if self.train else f'{model_type}_features_test.pt'

        feature_candidates = [
            os.path.join(_this_dir, features_filename),
            os.path.join(_this_dir, '..', features_filename),
        ]

        features_base = config.get("features_path")
        if features_base:
            feature_candidates.append(os.path.join(_resolve(features_base), features_filename))

        feature_path = next((p for p in feature_candidates if os.path.exists(p)), None)
        if feature_path is None:
            raise FileNotFoundError(f"Feature file not found: {features_filename} (searched {feature_candidates})")

        saved_features = torch.load(feature_path, map_location='cpu')
        self.text_features = saved_features['text_features']
        self.img_features = saved_features['img_features']

        self._select_features_by_classes()

    def _select_features_by_classes(self):
        """When a subset of classes is requested, filter features and remap labels."""
        if self.classes is None:
            self.class2local = None
            if self.train and self.images_per_class is None:
                # 默认保留训练集中每类 10 张图片
                self.images_per_class = 10
            elif not self.train and self.images_per_class is None:
                self.images_per_class = 1
            return

        cls_list = [int(c) for c in self.classes]
        self.classes = cls_list
        self.class2local = {orig: idx for idx, orig in enumerate(cls_list)}
        self.n_cls = len(cls_list)

        if self.train:
            if self.text_features.dim() != 2 or self.img_features.dim() != 2:
                raise ValueError(f"Unexpected feature shapes (train): text={self.text_features.shape}, img={self.img_features.shape}")
            n_all = 1654
            feature_dim = self.img_features.size(-1)
            try:
                img_reshaped = self.img_features.view(n_all, 10, feature_dim)
            except RuntimeError as err:
                raise RuntimeError(f"Failed to reshape training image features to [1654,10,D]; got {self.img_features.shape}") from err
            index = torch.tensor(cls_list, dtype=torch.long)
            selected = img_reshaped.index_select(0, index)
            if self.anchor_mode == 'first_per_class':
                selected = selected[:, :1, :]
                self.images_per_class = 1
            else:
                self.images_per_class = selected.size(1)
            self.text_features = self.text_features.index_select(0, index)
            self.img_features = selected.reshape(-1, feature_dim)
        else:
            index = torch.tensor(cls_list, dtype=torch.long)
            self.text_features = self.text_features.index_select(0, index)
            self.img_features = self.img_features.index_select(0, index)
            if self.images_per_class is None:
                self.images_per_class = 1

        if self.labels.dtype != torch.long:
            self.labels = self.labels.long()
        # Remap labels to local indices
        remapped = [self.class2local[int(val.item())] for val in self.labels]
        self.labels = torch.tensor(remapped, dtype=torch.long)
            
    def load_data(self):
        data_list = []
        label_list = []
        texts = []
        images = []
        times = None
        ch_names = None
        
        if self.train:
            directory = img_directory_training
        else:
            directory = img_directory_test
        
        dirnames = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
        dirnames.sort()
        
        if self.classes is not None:
            dirnames = [dirnames[i] for i in self.classes]

        for dir in dirnames:
            
            try:
                idx = dir.index('_')
                description = dir[idx+1:]  
            except ValueError:
                print(f"Skipped: {dir} due to no '_' found.")
                continue
                
            new_description = f"This picture is {description}"
            texts.append(new_description)

        if self.train:
            img_directory = img_directory_training  
        else:
            img_directory = img_directory_test
        
        all_folders = [d for d in os.listdir(img_directory) if os.path.isdir(os.path.join(img_directory, d))]
        all_folders.sort()  

        if self.classes is not None and self.pictures is not None:
            images = []  
            for i in range(len(self.classes)):
                class_idx = self.classes[i]
                pic_idx = self.pictures[i]
                if class_idx < len(all_folders):
                    folder = all_folders[class_idx]
                    folder_path = os.path.join(img_directory, folder)
                    all_images = [img for img in os.listdir(folder_path) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
                    all_images.sort()
                    if pic_idx < len(all_images):
                        images.append(os.path.join(folder_path, all_images[pic_idx]))
        elif self.classes is not None and self.pictures is None:
            images = []  
            for i in range(len(self.classes)):
                class_idx = self.classes[i]
                if class_idx < len(all_folders):
                    folder = all_folders[class_idx]
                    folder_path = os.path.join(img_directory, folder)
                    all_images = [img for img in os.listdir(folder_path) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
                    all_images.sort()
                    if self.anchor_mode == 'first_per_class' and self.train:
                        if all_images:
                            images.append(os.path.join(folder_path, all_images[0]))
                    else:
                        images.extend(os.path.join(folder_path, img) for img in all_images)
        elif self.classes is None:
            images = []  
            for folder in all_folders:
                folder_path = os.path.join(img_directory, folder)
                all_images = [img for img in os.listdir(folder_path) if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
                all_images.sort()  
                if self.anchor_mode == 'first_per_class' and self.train:
                    if all_images:
                        images.append(os.path.join(folder_path, all_images[0]))
                else:
                    images.extend(os.path.join(folder_path, img) for img in all_images)
        else:
            
            print("Error")
            
        print("self.subjects", self.subjects)
        print("exclude_subject", self.exclude_subject)
        subject_list = []
        for subject in self.subjects:
            subject_idx = self.subject_to_idx.get(subject)
            if subject_idx is None:
                continue
            if self.train:
                if subject == self.exclude_subject:  
                    continue            
                file_name = 'preprocessed_eeg_training.npy'

                file_path = os.path.join(self.data_path, subject, file_name)
                data = np.load(file_path, allow_pickle=True)
                
                preprocessed_eeg_data = torch.from_numpy(data['preprocessed_eeg_data']).float().detach()                
                times = torch.from_numpy(data['times']).detach()[50:]
                ch_names = data['ch_names']  

                n_classes = 1654  
                samples_per_class = 10  

                if self.classes is not None and self.pictures is not None:
                    for c, p in zip(self.classes, self.pictures):
                        start_index = c * samples_per_class + p
                        if start_index < len(preprocessed_eeg_data) and p < samples_per_class:  
                            preprocessed_eeg_data_class = preprocessed_eeg_data[start_index: start_index+1]  
                            labels = torch.full((1,), c, dtype=torch.long).detach()  
                            subjects_tensor = torch.full((1,), subject_idx, dtype=torch.long)
                            data_list.append(preprocessed_eeg_data_class)
                            label_list.append(labels)  
                            subject_list.append(subjects_tensor)
                            if self.images_per_class is None:
                                self.images_per_class = preprocessed_eeg_data_class.shape[0]

                elif self.classes is not None and self.pictures is None:
                    for c in self.classes:
                        start_index = c * samples_per_class
                        if self.anchor_mode == 'first_per_class':
                            preprocessed_eeg_data_class = preprocessed_eeg_data[start_index: start_index+1]
                        else:
                            preprocessed_eeg_data_class = preprocessed_eeg_data[start_index: start_index+samples_per_class]
                        sample_count = preprocessed_eeg_data_class.shape[0]
                        labels = torch.full((sample_count,), c, dtype=torch.long).detach()  
                        subjects_tensor = torch.full((sample_count,), subject_idx, dtype=torch.long)
                        data_list.append(preprocessed_eeg_data_class)
                        label_list.append(labels)
                        subject_list.append(subjects_tensor)
                        if self.images_per_class is None:
                            self.images_per_class = sample_count

                else:
                    for i in range(n_classes):
                        start_index = i * samples_per_class
                        if self.anchor_mode == 'first_per_class':
                            preprocessed_eeg_data_class = preprocessed_eeg_data[start_index: start_index+1]
                        else:
                            preprocessed_eeg_data_class = preprocessed_eeg_data[start_index: start_index+samples_per_class]
                        sample_count = preprocessed_eeg_data_class.shape[0]
                        labels = torch.full((sample_count,), i, dtype=torch.long).detach()  
                        subjects_tensor = torch.full((sample_count,), subject_idx, dtype=torch.long)
                        data_list.append(preprocessed_eeg_data_class)
                        label_list.append(labels)
                        subject_list.append(subjects_tensor)
                        if self.images_per_class is None:
                            self.images_per_class = sample_count

            else:
                if subject == self.exclude_subject or self.exclude_subject==None:  
                    file_name = 'preprocessed_eeg_test.npy'
                    file_path = os.path.join(self.data_path, subject, file_name)
                    data = np.load(file_path, allow_pickle=True)
                    preprocessed_eeg_data = torch.from_numpy(data['preprocessed_eeg_data']).float().detach()
                    times = torch.from_numpy(data['times']).detach()[50:]
                    ch_names = data['ch_names']  
                    n_classes = 200  # Each class contains 1 images
                    
                    samples_per_class = 1  

                    for i in range(n_classes):
                        if self.classes is not None and i not in self.classes:  
                            continue
                        start_index = i * samples_per_class  
                        preprocessed_eeg_data_class = preprocessed_eeg_data[start_index:start_index+samples_per_class]
                        labels = torch.full((samples_per_class,), i, dtype=torch.long).detach()  
                        preprocessed_eeg_data_class = torch.mean(preprocessed_eeg_data_class.squeeze(0), 0)
                        data_list.append(preprocessed_eeg_data_class)
                        label_list.append(labels)  
                        subject_list.append(torch.full((samples_per_class,), subject_idx if subject_idx is not None else -1, dtype=torch.long))
                        if not self.train and self.images_per_class is None:
                            self.images_per_class = samples_per_class
                else:
                    continue
        # datalist: (subjects * classes) * (10 * 4 * 17 * 100)
        # data_tensor: (subjects * classes * 10 * 4) * 17 * 100
        # data_list = np.mean(data_list, )
        # print("data_list", len(data_list))
        if self.images_per_class is None:
            if self.train and self.anchor_mode == 'first_per_class':
                self.images_per_class = 1
            elif self.train:
                self.images_per_class = 10
            else:
                self.images_per_class = 1

        if self.train:
            # print("data_list", *data_list[0].shape[1:])            
            data_tensor = torch.cat(data_list, dim=0).view(-1, *data_list[0].shape[2:])                 
            # data_tensor = torch.cat(data_list, dim=0).view(-1, *data_list[0].shape[1:])
            # data_tensor = torch.cat(data_list, dim=0).view(-1, *data_list[0].shape)   
            # print("label_tensor", label_tensor.shape)
            print("data_tensor", data_tensor.shape)
        else:           
            data_tensor = torch.cat(data_list, dim=0).view(-1, *data_list[0].shape)   
            # label_tensor = torch.cat(label_list, dim=0)
            # print("label_tensor", label_tensor.shape)
            # data_tensor = torch.cat(data_list, dim=0).view(-1, *data_list[0].shape[2:])
        # print("data_tensor", data_tensor.shape)
        # label_list: (subjects * classes) * 10
        # label_tensor: (subjects * classes * 10)
        # print("label_tensor = torch.cat(label_list, dim=0)")
        # print(label_list)
        label_tensor = torch.cat(label_list, dim=0)
        # label_tensor = torch.cat(label_list, dim=0)
        # print(label_tensor[:300])
        subject_tensor = torch.cat(subject_list, dim=0) if subject_list else None

        if self.train:
            # label_tensor: (subjects * classes * 10 * 4)
            label_tensor = label_tensor.repeat_interleave(4)
            if subject_tensor is not None:
                subject_tensor = subject_tensor.repeat_interleave(4)

        else:
            # label_tensor = label_tensor.repeat_interleave(80)
            # if self.classes is not None:
            #     unique_values = torch.unique(label_tensor, sorted=False)
           
            #     mapping = {val.item(): index for index, val in enumerate(torch.flip(unique_values, [0]))}
            #     label_tensor = torch.tensor([mapping[val.item()] for val in label_tensor], dtype=torch.long)
            pass      

                    
        self.times = times
        self.ch_names = ch_names
        self.sample_subject_indices = subject_tensor

        print(f"Data tensor shape: {data_tensor.shape}, label tensor shape: {label_tensor.shape}, text length: {len(texts)}, image length: {len(images)}")
        
        return data_tensor, label_tensor, texts, images

    def extract_eeg(self, eeg_data, time_window):

        start, end = time_window

        # Get the indices of the times within the specified window
        indices = (self.times >= start) & (self.times <= end)
        # print("self.times", self.times.shape)
        # print("indices", indices)
        # print("indices", indices.shape)
        # print("eeg_data", eeg_data.shape)
        # Use these indices to select the corresponding data
        extracted_data = eeg_data[..., indices]
        # print(f"extracted_data shape: {extracted_data.shape}")

        return extracted_data
    
    def __getitem__(self, index):
        # Get the data and label corresponding to "index"
        # index: (subjects * classes * 10 * 4)
        x = self.data[index]
        label = self.labels[index]
        
        if self.pictures is None:
            if self.classes is None:
                index_n_sub_test = self.n_cls * 1 * 80
                index_n_sub_train = self.n_cls * self.images_per_class * 4
            else:
                index_n_sub_test = len(self.classes)* 1 * 80
                index_n_sub_train = len(self.classes)* self.images_per_class * 4
            # text_index: classes
            if self.train:
                text_index = (index % index_n_sub_train) // (self.images_per_class * 4)
            else:
                text_index = (index % index_n_sub_test)
            # img_index: classes * 10
            if self.train:
                img_index = (index % index_n_sub_train) // (4)
            else:
                img_index = (index % index_n_sub_test)
        else:
            if self.classes is None:
                index_n_sub_train = self.n_cls * 1 * 4
                index_n_sub_test = self.n_cls * 1 * 80
            else:
                index_n_sub_test = len(self.classes)* 1 * 80
                index_n_sub_train = len(self.classes)* 1 * 4
            # text_index: classes
            if self.train:
                text_index = (index % index_n_sub_train) // (1 * 4)
            else:
                text_index = (index % index_n_sub_test)
            # img_index: classes * 10
            if self.train:
                img_index = (index % index_n_sub_train) // (4)
            else:
                img_index = (index % index_n_sub_test)
        # print("text_index", text_index)
        # print("self.text", self.text)
        # print("self.text", len(self.text))
        text = self.text[text_index]
        img = self.img[img_index]
        
        text_features = self.text_features[text_index]
        img_features = self.img_features[img_index]
        sample = (x, label.long(), text, text_features, img, img_features)
        if self.return_subject_ids:
            if self.sample_subject_indices is None:
                raise RuntimeError("sample_subject_indices is not available; enable return_subject_ids only when supported.")
            subject_idx = self.sample_subject_indices[index]
            sample = sample + (subject_idx.long(),)

        return sample

    def __len__(self):
        return self.data.shape[0]  # or self.labels.shape[0] which should be the same

if __name__ == "__main__":
    # Instantiate the dataset and dataloader
    # data_path = "/home/ldy/Workspace/THINGS/EEG/osfstorage-archive"  # Replace with the path to your data
    data_path = data_path
    train_dataset = EEGDataset(data_path, subjects = ['sub-01'], train=True)    
    test_dataset = EEGDataset(data_path, subjects = ['sub-01'], train=False)
    # train_dataset = EEGDataset(data_path, exclude_subject = 'sub-01', train=True)    
    # test_dataset = EEGDataset(data_path, exclude_subject = 'sub-01', train=False)    
    # train_dataset = EEGDataset(data_path, train=True) 
    # test_dataset = EEGDataset(data_path, train=False) 
    
    
    
    
    # 100 Hz
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
    
    i = 80*1-1
    x, label, text, text_features, img, img_features  = test_dataset[i]
    print(f"Index {i}, Label: {label}, text: {text}")
    Image.open(img)
            
    
        
    