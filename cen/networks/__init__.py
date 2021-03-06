from . import densenet
from . import resnet
from . import rnn
from . import simple
from . import vgg16
from . import inceptionv3

def get(name, **kwargs):
    if name == "bert":
        return rnn.Bert(**kwargs)
    elif name == "bilstm":
        return rnn.BiLSTM(**kwargs)
    elif name == "densenet":
        return densenet.DenseNet(**kwargs)
    elif name == "resent50":
        return resnet.ResNet50(**kwargs)
    elif name == "simple_cnn":
        return simple.CNN(**kwargs)
    elif name == "simple_mlp":
        return simple.MLP(**kwargs)
    elif name == "identity":
        return simple.Identity()
    elif name == "Inception_v3":
        return inceptionv3.Inception_v3(**kwargs)
    else:
        raise ValueError(f"Unknown network: {name}")
