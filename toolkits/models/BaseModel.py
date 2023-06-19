import os
import torch
import torch.nn as nn
from pytorch_pretrained.file_utils import WEIGHTS_NAME
from tools.common import logger


class BaseModel(nn.Module):
    def __init__(self, config):
        super(BaseModel, self).__init__()
        self.config = config
        self.num_labels = config.num_labels

    def init_weights(self, method='xavier', exclude='embedding'):
        for name, w in self.named_parameters():
            if exclude not in name:
                if 'weight' in name:
                    if method == 'xavier':
                        nn.init.xavier_normal_(w)
                    elif method == 'kaiming':
                        nn.init.kaiming_normal_(w)
                    else:
                        nn.init.normal_(w)
                elif 'bias' in name:
                    nn.init.constant_(w, 0)
                else:
                    pass

    def save_pretrained(self, save_directory):
        """ Save a model and its configuration file to a directory, so that it
            can be re-loaded using the `:func:`~pytorch_transformers.PreTrainedModel.from_pretrained`` class method.
        """
        assert os.path.isdir(save_directory), "Saving path should be a directory where the model and configuration can be saved"

        # Only save the model it-self if we are using distributed training
        model_to_save = self.module if hasattr(self, 'module') else self

        # Save configuration file
        model_to_save.config.save_pretrained(save_directory)

        # If we save using the predefined names, we can load using `from_pretrained`
        output_model_file = os.path.join(save_directory, WEIGHTS_NAME)

        torch.save(model_to_save.state_dict(), output_model_file)

    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        if os.path.isdir(pretrained_model_name_or_path):
            archive_file = os.path.join(pretrained_model_name_or_path, WEIGHTS_NAME)
        else:
            archive_file = pretrained_model_name_or_path

        model_to_load = cls.module if hasattr(cls, 'module') else cls
        if os.path.split(archive_file)[1] != WEIGHTS_NAME or not os.path.exists(archive_file):
            return model_to_load

        logger.info("loading weights file {}".format(archive_file))
        model_to_load.load_state_dict(torch.load(archive_file, map_location='cpu'))
        model_to_load.eval()
        return model_to_load
