import numpy as np
import tensorflow as tf
import torch

def set_random_seed(auto_var):
    random_seed = auto_var.get_var("random_seed")

    torch.manual_seed(random_seed)
    tf.random.set_seed(random_seed)
    np.random.seed(random_seed)

    random_state = np.random.RandomState(auto_var.get_var("random_seed"))
    auto_var.set_intermidiate_variable("random_state", random_state)

    return random_state