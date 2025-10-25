# Cheatsheet ML Lab

## numpy

Create numpy array of shape (1,2,3) and type uint8 contiaining ones.
```python3
arr = numpy.ones(shape=(1,2,3), dtype=numpy.uint8)
```

Numpy array access.
```python3
arr[0] # first row
arr[:, 0] # first col
```

Collapse numpy array along axis.
```python3
np.sum(arr, axis=0)   # collapse first dimension
```

Sample from normal distribution, where: loc ~ mean, scale ~ std, size ~ shape

```python3
np.random.normal(loc=0.0, scale=1.0, size=(130,2))
```

## matplotlib

```python3
fig, ax = plt.subplots(1,1,figsize=(7,7))
```

Plot points onto plot.
```python3
ax.scatter(x=arr[0], y=arr[1], c='r') # first rows x's and y's, color red
```

Add grid and cutoff empty scpaces
```python3
fig.tight_layout()
plt.grid()
```

### TODO: Add week 2 and following
