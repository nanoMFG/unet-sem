# Python
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.experimental import RMSprop
from unetSem.models.unet import unet  # replace with the actual module where unet is defined

def test_unet():
    model = unet(input_size=(256, 256, 3), num_classes=2, learning_rate=0.001, optimizer='RMSprop')

    # Check the number of output classes
    assert model.output_shape[-1] == 2

    # Check the optimizer
    assert isinstance(model.optimizer, RMSprop)

    # Check the learning rate
    assert model.optimizer.learning_rate == 0.001

    # Test with a different optimizer
    model = unet(input_size=(256, 256, 3), num_classes=3, learning_rate=0.01, optimizer='Adam')

    # Check the number of output classes
    assert model.output_shape[-1] == 3

    # Check the optimizer
    assert isinstance(model.optimizer, Adam)

    # Check the learning rate
    assert model.optimizer.learning_rate == 0.01

    # Test with an invalid optimizer
    with pytest.raises(ValueError):
        model = unet(input_size=(256, 256, 3), num_classes=2, learning_rate=0.001, optimizer='InvalidOptimizer')