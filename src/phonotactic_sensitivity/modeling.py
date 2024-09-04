import warnings
import torch

import numpy as np

from collections import defaultdict

from transformers import (
    logging,
    AutoConfig,
    Wav2Vec2ForCTC,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Model,
)

from phonotactic_sensitivity.paths import ROOT
from phonotactic_sensitivity.settings import models


class SaveOutput:
    def __init__(self):
        self.outputs = defaultdict()

    def __call__(self, name):
        def hook(module, module_in, module_out):
            self.outputs[name] = module_out.detach()

        return hook

    def clear(self):
        self.outputs = defaultdict()


class AudioModel:
    _supported_models = [
        "w2v2_base_unt",
        "w2v2_base_pret-sp",
        "w2v2_base_pret-acs",
        "w2v2_base_ft",
        "w2v2_large_unt",
        "w2v2_large_pret-sp",
        "w2v2_large_ft",
    ]

    def __init__(self, model_identifier: str, debug=False):
        assert model_identifier in self._supported_models, f"model_identifier must be one of {self._supported_models}"

        self._debug_mode = debug
        if not self._debug_mode:
            logging.set_verbosity(logging.CRITICAL)
            warnings.filterwarnings(action='ignore', category=UserWarning)

        if model_identifier.endswith("unt"):
            config = AutoConfig.from_pretrained(models[model_identifier]['model_config_name_or_path'])
            self.model = Wav2Vec2Model(config)
        elif model_identifier.endswith("acs"):
            config = AutoConfig.from_pretrained(ROOT / models[model_identifier]['model_config_name_or_path'])
            self.model = Wav2Vec2Model.from_pretrained(
                ROOT / models[model_identifier]['pretrained_model_name_or_path'], config=config
            )
        elif model_identifier.endswith("ft"):
            config = AutoConfig.from_pretrained(models[model_identifier]['model_config_name_or_path'])
            self.model = Wav2Vec2ForCTC.from_pretrained(
                models[model_identifier]['pretrained_model_name_or_path'], config=config
            )
        else:
            config = AutoConfig.from_pretrained(models[model_identifier]['model_config_name_or_path'])
            self.model = Wav2Vec2Model.from_pretrained(
                models[model_identifier]['pretrained_model_name_or_path'], config=config
            )

        self.feature_extractor = Wav2Vec2FeatureExtractor()

        if type(self.model) == Wav2Vec2ForCTC:
            tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(
                self.model.config._name_or_path, clean_up_tokenization_spaces=False
            )
            lm_head = self.model.lm_head
            self.ctc_components = [lm_head, tokenizer]

    def get_activations(self, audio_signals: list, samp_freq: int):
        """
        Extract hidden states for the given audio_signals.
        """
        # register hooks
        save_output = SaveOutput()

        if type(self.model) == Wav2Vec2ForCTC:
            last_conv_layer = self.model.wav2vec2.feature_extractor.conv_layers[-1]
            last_conv_layer.activation.register_forward_hook(save_output('CNN'))
            self.model.wav2vec2.encoder.layer_norm.register_forward_hook(save_output('embeds'))
            for i, enc_layer in enumerate(self.model.wav2vec2.encoder.layers):
                enc_layer.final_layer_norm.register_forward_hook(save_output(f'T{i+1}'))
        elif type(self.model) == Wav2Vec2Model:
            last_conv_layer = self.model.feature_extractor.conv_layers[-1]
            last_conv_layer.activation.register_forward_hook(save_output('CNN'))
            self.model.encoder.layer_norm.register_forward_hook(save_output('embeds'))
            for i, enc_layer in enumerate(self.model.encoder.layers):
                enc_layer.final_layer_norm.register_forward_hook(save_output(f'T{i+1}'))
        else:
            raise NotImplementedError

        inputs = self.feature_extractor(
            audio_signals, sampling_rate=samp_freq, padding=True, return_tensors="pt"
        ).input_values

        # forward pass
        self.model.eval()
        with torch.no_grad():
            self.model(inputs)

        for layer in save_output.outputs.keys():
            if layer.startswith('C'):
                save_output.outputs[layer] = save_output.outputs[layer]
            else:
                save_output.outputs[layer] = save_output.outputs[layer]

        saved_output = dict(save_output.outputs)

        return saved_output

    def intv_times_to_frames(self, intv_times: tuple[float, float]):
        """
        Convert interval start & end time (in seconds) to start & end frame indices.
        """
        start_time, end_time = intv_times
        frame_rate = self.model.config.inputs_to_logits_ratio / self.feature_extractor.sampling_rate
        start_frame = int(np.floor(start_time / frame_rate))
        end_frame = int(np.ceil(end_time / frame_rate))
        return (start_frame, end_frame)


def mean_step_embeddings(step_activations, from_frame: int = 0, to_frame: int = -1):
    step_embeddings = {}
    for layer in step_activations.keys():
        if layer == 'CNN':
            step_embeddings[layer] = step_activations[layer][:, :, from_frame:to_frame].mean(axis=2)
        else:
            step_embeddings[layer] = step_activations[layer][:, from_frame:to_frame, :].mean(axis=1)
    return step_embeddings
