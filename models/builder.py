import warnings
from tools.registry import Registry

MODELS = Registry('models')

LEFT = MODELS
RIGHT = MODELS
CLASSIFIERS = MODELS

def build_left(cfg):
    """Build left model."""
    return LEFT.build(cfg)


def build_right(cfg):
    """Build right model."""
    return RIGHT.build(cfg)


def build_classifier(cfg):
    """Build classifier."""
    # if train_cfg is not None or test_cfg is not None:
    #     warnings.warn(
    #         'train_cfg and test_cfg is deprecated, '
    #         'please specify them in model', UserWarning)
    # assert cfg.get('train_cfg') is None or train_cfg is None, \
    #     'train_cfg specified in both outer field and model field '
    # assert cfg.get('test_cfg') is None or test_cfg is None, \
    #     'test_cfg specified in both outer field and model field '
    return CLASSIFIERS.build(cfg)
