- [Other notes](#other-notes)
  - [Torch tensors](#torch-tensors)

# Other notes
## Torch tensors

Create torch tensor from python data

```
some_integers = torch.tensor((2, 3, 5, 7, 11, 13, 17, 19))
print(some_integers)
```


This does not copy, it is just a label
```
a = torch.ones(2, 2)
b = a
```

To copy:
```
a = torch.ones(2, 2)
b = a.clone()
```

Moving to CPU, GPU

- To do computing tensors must be in the same device

```
if torch.cuda.is_available():
    print('We have a GPU!')
else:
    print('Sorry, CPU only.')
```


```
device = torch.device("cpu")

```

```
if torch.cuda.is_available():
    my_device = torch.device('cuda')
else:
    my_device = torch.device('cpu')
print('Device: {}'.format(my_device))

x = torch.rand(2, 2, device=my_device)
print(x)
```


Moving to another device
```
y = torch.rand(2, 2)
y = y.to(my_device)
``` 
Devices:
- meta device: Tensor without any data attached to it