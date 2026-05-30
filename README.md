## Overview

This project is an image classification training pipeline built with PyTorch.

## Features

- Uses a pretrained ResNet50 model
- Trains the final fully connected layer
- Saves the best model checkpoint

## Project Structure

```text
Visualization/
├── Main.py
├── DataBuilder.py
├── ModelBuilder.py
├── Trainer.py
└── Validator.py

## Performance Optimization

This project uses num_workers=2 and pin_memory=True in the DataLoader to reduce GPU waiting time during data loading. It also uses non_blocking=True when moving batches to the GPU.
