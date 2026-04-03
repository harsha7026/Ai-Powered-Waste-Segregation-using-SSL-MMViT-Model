# Place your trained model weights here

Example:
- waste_classifier.pt (trained model)
- waste_classifier_v2.pt (improved model)

The model file should be a PyTorch state dict saved with:
```python
torch.save(model.state_dict(), 'waste_classifier.pt')
```

Default model path is configured in `app/config.py`.
