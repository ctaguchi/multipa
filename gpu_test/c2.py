import torch

print('cuda.is_available:    ', torch.cuda.is_available())
print('cuda.device_count:    ', torch.cuda.device_count())
print('cuda.current_device:  ', torch.cuda.current_device())
print('get_device_name:      ', torch.cuda.get_device_name(torch.cuda.current_device()))

tensor = torch.randn(2, 2)
res = tensor.to(0)
print(res)


