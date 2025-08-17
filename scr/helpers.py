import os
import torch
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader
import random
import matplotlib.pyplot as plt
# import sys
# sys.path.append('../')

def steering_vector(N, deg):
    """
    Calculate the steering vector for a uniform linear array using given antenna configuration.
    
    Args:
        N (int): Number of antenna elements.
        deg (float): Angle of arrival in degrees.

    Returns:
        torch.Tensor: The steering vector as a complex-valued tensor.
    """
    d = 0.5  # Element spacing (in units of wavelength)
    wavelength = 1.0  # Wavelength of the signal (same units as d)
    k = 2 * torch.pi / wavelength  # Wavenumber
    n = torch.arange(0, N).view(N, 1)  # Antenna element indices [0, 1, ..., N-1]
    theta = deg * torch.pi / 180  # Convert degrees to radians
    phases = k * d * n * torch.sin(theta)  # Phase shift for each element

    return torch.exp(1j * phases)  # Complex exponential for each phase shift



def generate_complex_signal(N=10, snr_db=10, deg=torch.tensor([30]),amp=torch.tensor([1])):
    """
    Generates a complex-valued signal for an array of N antenna elements.

    Args:
        N (int): Number of antenna elements.
        snr_db (float): Signal-to-Noise Ratio in decibels.
        deg (tensor): Angle of arrival in degrees.

    Returns:
        torch.Tensor: Complex-valued tensor of shape (N, 1) representing the received signals.
    """
    a_theta = steering_vector(N, deg)
    phase = (amp * torch.exp(2j * torch.pi * torch.randn(a_theta.size()[1]))).view(-1, 1)
    signal = torch.matmul(a_theta.to(phase.dtype), phase)
    # signal_power = torch.mean(torch.abs(signal)**2)
    signal_power = torch.min(amp)**2
    snr_linear = 10**(snr_db / 10)

    noise_power = signal_power / snr_linear
    noise_real = torch.sqrt(noise_power / 2) * torch.randn_like(signal.real)
    noise_imag = torch.sqrt(noise_power / 2) * torch.randn_like(signal.imag)
    noise = torch.complex(noise_real, noise_imag)

    return signal + noise 


def generate_label(degrees, min_angle=-30, max_angle=30):
    """
    Generate one-hot encoded labels for the given degrees.
    
    Args:
        degrees (tensor): Target angles in degrees.

    Returns:
        torch.Tensor: One-hot encoded labels.
    """
    labels = torch.zeros(max_angle - min_angle + 1)
    indices = degrees - min_angle
    labels[indices.long()] = 1
    return labels

def generate_data(N, num_samples=1, max_targets=3, folder_path='/content/drive/MyDrive/Asilomar2024/data/'):
    """
    Generate dataset with random number of targets and varying SNR levels.
    
    Args:
        N (int): Number of antenna elements.
        num_samples (int): Number of samples to generate for each SNR level.
        max_targets (int): Maximum number of targets.
        folder_path (str): Base folder path for saving data.

    Returns:
        int: Always returns 0. Data saved in specified directory.
    """
    angles = torch.arange(-30, 31, 1)
    signal_folder = os.path.join(folder_path, 'signal')
    label_folder = os.path.join(folder_path, 'label')
    os.makedirs(signal_folder, exist_ok=True)
    os.makedirs(label_folder, exist_ok=True)

    for snr_db in tqdm(range(0, 35, 5), desc='SNR levels', unit='snr', dynamic_ncols=True):
        all_signals, all_labels = [], []
        for _ in range(num_samples):
            num_targets = torch.randint(1, max_targets + 1, (1,)).item()
            deg_indices = torch.randperm(len(angles))[:num_targets]
            degs = angles[deg_indices]
            label = generate_label(degs)
            noisy_signal = generate_complex_signal(N=N, snr_db=snr_db, deg=degs)
            all_signals.append(noisy_signal)
            all_labels.append(label)
        torch.save(all_signals, os.path.join(signal_folder, f'signals_snr_{snr_db}dB.pt'))
        torch.save(all_labels, os.path.join(label_folder, f'labels_snr_{snr_db}dB.pt'))
    return None 
 
def randSparse(signal,sparsity):
    sparseSignal = signal.clone()
    sparseInd = torch.randperm(signal.shape[-1])[:int(signal.shape[-1] * sparsity)]
    sparseSignal[:,:,sparseInd] = 0
    return sparseSignal

class SignalDataset(Dataset):
    def __init__(self, file_paths,transform=None):
        """
        Initializes a dataset containing signals and their corresponding labels.

        Args:
            file_paths (list): Paths to files containing signals.
            label_paths (list): Paths to files containing labels.
        """
        self.signals = [torch.load(file)['data'] for file in file_paths]
        self.labels = [torch.load(file)['label'] for file in file_paths]
        self.alphas = [torch.load(file)['alpha'] for file in file_paths]
        self.filename = [file for file in file_paths]
        self.transform = transform
        # self.iaa = [torch.load(file)['iaa'] for file in file_paths]
        # self.signals = torch.cat(self.signal, dim=0)
        # self.labels = torch.cat(self.label, dim=0)
        # self.alphas = torch.cat(self.alpha, dim=0)

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        signal = self.signals[idx]
        label = self.labels[idx]
        alpha = self.alphas[idx]
        filename = self.filename[idx]
        if self.transform:
            signal = torch.view_as_complex(signal)
            signal, label = self.transform(signal, label)
            signal = torch.view_as_real(signal)
        return signal,label,alpha
# 0:6400
def custom_transform(signal, label):
    # Apply transformations conditionally
    # signal, label = flip_transform(signal, label)
    # signal = add_noise_transform(signal)
    # signal, label = shift_transform(signal, label)
    if torch.rand(1) > 0.5:
        signal, label = flip_transform(signal, label)
    if torch.rand(1) > 0.5:
        signal, label = shift_transform(signal, label)
    # if torch.rand(1) > 0.5:
    #     signal = add_noise_transform(signal)
    return signal, label

def flip_transform(signal, label):
    signal_real = signal.real
    signal_complex = -1 * signal.imag
    signal = torch.complex(signal_real,signal_complex)
    label = torch.flip(label, dims=[1])
    return signal, label

def shift_transform(signal, label):
    # Random shift parameters
    output_size = 100
    number_element = signal.shape[1]
    shift_idx = torch.randperm(output_size)[0]
    random_signs = torch.randint(0, 2, (1,)) * 2 - 1
    shift_idx = shift_idx * random_signs
    
    linear_space = torch.linspace(-1, 1, 256)
    grid_size = linear_space[1] - linear_space[0]
    shift_freq = grid_size * shift_idx
    shift_deg = torch.asin(shift_freq) * (180 / torch.pi)
    bv_shift = steering_vector(number_element, shift_deg).T
    signal_new = bv_shift * signal
    label_new0 = torch.roll(label[:,1:], shift_idx.item(), dims=1)
    label_new1 = label_new0[:,-1].unsqueeze(-1)
    label_new = torch.cat((label_new1, label_new0),dim=1)
    return signal_new, label_new

def modify_label_for_shift(label, h_shift, v_shift):
    # Implement how label should change on shift if necessary
    return label


def create_dataloader(data_path, batch_size=32, shuffle=True, transform=False):
    """
    Create a DataLoader for batching and shuffling the dataset.

    Args:
        data_path (str): Path to the directory containing the data files.
        batch_size (int): Number of samples per batch.
        shuffle (bool): Whether to shuffle the data.

    Returns:
        DataLoader: Configured DataLoader for the dataset.
    """
    data_files = [os.path.join(data_path, f) for f in os.listdir(data_path)]
    if transform:  
        dataset = SignalDataset(sorted(data_files),transform = custom_transform)
    else:
        dataset = SignalDataset(sorted(data_files))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


class UnetDataset(Dataset):
    def __init__(self, file_paths,transform=None):
        """
        Initializes a dataset containing signals and their corresponding labels.

        Args:
            file_paths (list): Paths to files containing signals.
            label_paths (list): Paths to files containing labels.
        """
        self.signals = [torch.load(file)['data'] for file in file_paths]
        self.labels = [torch.load(file)['label'] for file in file_paths]
        self.filename = [file for file in file_paths]
        self.transform = transform

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, idx):
        signal = self.signals[idx]
        label = self.labels[idx] 
        filename = self.filename[idx]
        if self.transform:
            signal, label = self.transform(signal, label)
        return signal,label#,filename

def unet_transform(signal, label):
    # Apply transformations conditionally
    if torch.rand(1) > 0.5:
        signal, label = unet_flip_transform(signal, label)
    if torch.rand(1) > 0.5:
        signal, label = unet_shift_transform(signal, label)
    return signal, label

def unet_flip_transform(signal, label):
    if len(signal.shape) == 2:
        signal = torch.flip(signal, [1])
        label = torch.flip(label, [1])
    else: 
        signal = torch.flip(signal, [2])
        label = torch.flip(label, [2])            
    return signal, label

def unet_shift_transform(signal, label):
    # Random shift parameters
    shift_size = 256
    random_shift = random.randint(0, shift_size - 1)
    if len(signal.shape) == 2:
        signal = torch.roll(signal,random_shift, [1])
        label = torch.roll(label,random_shift, [1])
    else: 
        signal = torch.roll(signal,random_shift, [2])
        label = torch.roll(label,random_shift, [2])  
    return signal, label


def create_dataloader_unet(data_path, batch_size=32, shuffle=True, transform=False):
    """
    Create a DataLoader for batching and shuffling the dataset.

    Args:
        data_path (str): Path to the directory containing the data files.
        batch_size (int): Number of samples per batch.
        shuffle (bool): Whether to shuffle the data.

    Returns:
        DataLoader: Configured DataLoader for the dataset.
    """
    data_files = [os.path.join(data_path, f) for f in os.listdir(data_path)]
    if transform:  
        dataset = UnetDataset(sorted(data_files),transform = unet_transform)
    else:
        dataset = UnetDataset(sorted(data_files))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def unwrap_batch(angles, discont=torch.pi):
    """
    Unwrap a batch of tensors of angles.

    Parameters:
    - angles : torch.Tensor, 2D tensor of angles with shape (batch_size, sequence_length)
    - discont : float, maximum discontinuity between consecutive angles, beyond which correction is applied

    Returns:
    - unwrapped_angles : torch.Tensor, 2D tensor of unwrapped angles with the same shape as input
    """
    # Compute the differences between consecutive angles along the last dimension
    d = torch.diff(angles, dim=1)

    # Identify where the jumps are greater than pi or less than -pi
    d_adj = torch.where(d > discont, d - 2 * torch.pi, d)
    d_adj = torch.where(d_adj < -discont, d_adj + 2 * torch.pi, d_adj)

    # Reconstruct the original angles adjusted for the discontinuities
    # We need to add a dimension of zeros at the start of d_adj to offset the diff operation
    d_adj_padded = torch.cat([torch.zeros((d_adj.shape[0], 1), device=d_adj.device, dtype=d_adj.dtype), d_adj], dim=1)
    unwrapped_angles = angles[:, 0:1] + torch.cumsum(d_adj_padded, dim=1)

    return unwrapped_angles

# train_loader = create_dataloader_unet(os.path.join('../data/BAMA/2d/', 'train'), batch_size=4,transform=True)
# val_loader = create_dataloader_unet(os.path.join('../data/BAMA/2d/', 'val'), batch_size=4,transform=True)
# for signals, labels in tqdm(val_loader):
#     print(signals.shape)
#     print(labels.shape)
#     print(alphas.shape)
    