import os
import logging
from transformers.trainer import Trainer
from transformers.file_utils import ModelOutput
import torch
from torch import Tensor
from dataclasses import dataclass
from typing import Union, Tuple, Optional
from src.model.bgem3 import M3DenseEmbedModel


@dataclass
class EncoderOutput(ModelOutput):
    q_reps: Optional[Tensor] = None
    p_reps: Optional[Tensor] = None
    loss: Optional[Tensor] = None
    scores: Optional[Tensor] = None


class CustomTrainer(Trainer):
    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logging.info("Saving model checkpoint to %s", output_dir)
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        if not hasattr(self.model, 'save'):
            raise NotImplementedError(
                f'MODEL {self.model.__class__.__name__} '
                f'does not support save interface')
        else:
            self.model.save(output_dir)
        if self.tokenizer is not None and self.is_world_process_zero():
            self.tokenizer.save_pretrained(output_dir)

        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))

    def compute_loss(self, model:M3DenseEmbedModel, inputs, return_outputs=False) -> Union[Tuple[Tensor, EncoderOutput], Tensor]:
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        loss = model(inputs)
        return (loss, EncoderOutput(loss=loss)) if return_outputs else loss
