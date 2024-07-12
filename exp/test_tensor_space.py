import torch
import gzip
import os

def save_tensor(tensor, file_path, compress=False):
	"""
	Save a PyTorch tensor to a file, with optional compression.

	Args:
		tensor (torch.Tensor): The tensor to save.
		file_path (str): Path to the file where the tensor will be saved.
		compress (bool): Whether to compress the saved tensor file.
	"""
	if compress:
		with gzip.open(file_path + '.gz', 'wb') as f:
			torch.save(tensor, f)
	else:
		torch.save(tensor, file_path)

def load_tensor(file_path, compressed=False):
	"""
	Load a PyTorch tensor from a file, with support for compressed files.

	Args:
		file_path (str): Path to the file where the tensor is saved.
		compressed (bool): Whether the tensor file is compressed.

	Returns:
		torch.Tensor: The loaded tensor.
	"""
	if compressed:
		with gzip.open(file_path, 'rb') as f:
			tensor = torch.load(f)
	else:
		tensor = torch.load(file_path)
	return tensor

# Example usage
tensor = torch.randn(10000, 256)  # Example tensor
save_path = 'dat/test_pt/tensor.pt'

# Save without compression
save_tensor(tensor, save_path, compress=False)

# Save with compression
save_tensor(tensor, save_path, compress=True)

# Load the tensor back
loaded_tensor = load_tensor(save_path + '.gz', compressed=True)