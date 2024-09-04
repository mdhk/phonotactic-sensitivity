import torch
import pandas as pd

from pathlib import Path

from phonotactic_sensitivity.paths import ROOT
from phonotactic_sensitivity.settings import models, voices, seed
from phonotactic_sensitivity.file_loading import load_continua
from phonotactic_sensitivity.modeling import AudioModel, mean_step_embeddings
from phonotactic_sensitivity.measures import EmbeddingSimilarity, CTCLens, ProbeProbability

torch.manual_seed(seed)

continua = load_continua(pd.read_csv(ROOT / 'data/continua.csv', sep=';'))

embsim_dir = Path(ROOT / 'results/embedding_similarities')
ctcprob_dir = Path(ROOT / 'results/ctc_probabilities')
pcprob_dir = Path(ROOT / 'results/probe_probabilities')

embsim_dir.mkdir(parents=True, exist_ok=True)
ctcprob_dir.mkdir(parents=True, exist_ok=True)
pcprob_dir.mkdir(parents=True, exist_ok=True)

for model_name in models.keys():
    model = AudioModel(model_name)

    if "base" in model_name:
        ctc_components = AudioModel("w2v2_base_ft").ctc_components
    elif "large" in model_name:
        ctc_components = AudioModel("w2v2_large_ft").ctc_components
    else:
        raise NotImplementedError

    embsims = []
    ctcprobs = []
    pcprobs = []

    for voice in voices.keys():

        for continuum_name in continua[voice].keys():

            continuum = continua[voice][continuum_name]
            step_activations = model.get_activations(continuum.steps, continuum._samp_freq)
            morph_start_frame, morph_end_frame = model.intv_times_to_frames(continuum.morph_interval)
            step_embeddings = mean_step_embeddings(
                step_activations, from_frame=morph_start_frame, to_frame=morph_end_frame
            )
            embsim = EmbeddingSimilarity()
            embsim_df = embsim.compute(step_embeddings).to_df()
            ctclens = CTCLens(*ctc_components)
            ctcprob_df = (
                ctclens.compute_logits(step_activations)
                .to_token_probs(from_frame=morph_start_frame, to_frame=morph_end_frame)
                .to_df()
            )
            pcprob = ProbeProbability(ROOT / f'models/probing_classifiers/lr-{model_name}.pkl')
            pcprob_df = pcprob.compute(step_embeddings).to_df()
            for df in [embsim_df, ctcprob_df, pcprob_df]:
                df['voice'] = voice
                df['continuum'] = continuum_name
            embsims.append(embsim_df)
            ctcprobs.append(ctcprob_df)
            pcprobs.append(pcprob_df)

    pd.concat(embsims).to_csv(embsim_dir / f'{model_name}.csv', sep=';', index=False)
    pd.concat(ctcprobs).to_csv(ctcprob_dir / f'{model_name}.csv', sep=';', index=False)
    pd.concat(pcprobs).to_csv(pcprob_dir / f'{model_name}.csv', sep=';', index=False)

print('Done! Results saved in:\n' + f'\t{embsim_dir}\n' + f'\t{ctcprob_dir}\n' + f'\t{pcprob_dir}')
