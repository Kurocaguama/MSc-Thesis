import torch
print(torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(torch.cuda.get_device_properties(i).name)
    
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(dev)