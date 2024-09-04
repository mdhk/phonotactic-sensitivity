import pprint
import librosa
import tgt

from pathlib import Path
from phonotactic_sensitivity.paths import ROOT
from phonotactic_sensitivity.settings import SAMP_FREQ


class Continuum:
    def __init__(self, file_dir: Path, endpoints: tuple[str, str], tokens: tuple[str, str], voice: str, samp_freq: int):

        self._file_dir = Path(file_dir)
        self._samp_freq = int(samp_freq)

        self.endpoint_A, self.endpoint_B = tuple(map(str, endpoints))
        self.token_A, self.token_B = tuple(map(str, tokens))
        self.voice = str(voice)

        tg_A = tgt.io.read_textgrid(self._file_dir / f"{self.endpoint_A}-{self.endpoint_B}_cntnm_000.TextGrid")
        tg_B = tgt.io.read_textgrid(self._file_dir / f"{self.endpoint_A}-{self.endpoint_B}_cntnm_010.TextGrid")

        self.morph_interval = self._get_morph_interval(tg_A, tg_B)
        self.steps = self._get_steps(self._file_dir / "steps")

    def __str__(self):
        return pprint.pformat(self.__dict__, sort_dicts=False)

    def __repr__(self):
        cwd_to_ROOT = '/'.join(len(Path.cwd().relative_to(ROOT).parts) * ['..']) + '/'
        repr_str = (
            f"Continuum(\n\t"
            + f"file_dir={(cwd_to_ROOT + str(self._file_dir.relative_to(ROOT)))!r},\n\t"
            + f"endpoints=({self.endpoint_A!r}, {self.endpoint_B!r})\n\t"
            + f"tokens=({self.token_A!r}, {self.token_B!r})\n\t"
            + f"voice={self.voice!r}\n\t)"
        )
        return repr_str

    def _get_morph_interval(self, tg_A: tgt.core.TextGrid, tg_B: tgt.core.TextGrid) -> tuple[float, float]:
        """
        Read TextGrid annotations of the target sound at the first and last continuum step,
        return the (start_time, end_time) interval (in seconds) of the morphing target sound.
        """
        morph_intv_A = tg_A.get_tier_by_name("morphtarget").get_annotations_with_text("X")[0]
        morph_intv_B = tg_B.get_tier_by_name("morphtarget").get_annotations_with_text("X")[0]
        morph_intv_union = tuple(
            map(
                float,
                (
                    min(morph_intv_A.start_time, morph_intv_B.start_time),
                    max(morph_intv_A.end_time, morph_intv_B.end_time),
                ),
            )
        )
        return morph_intv_union

    def _get_steps(self, steps_dir: Path) -> list:
        """
        Load audio timeseries for each step in the continuum.
        """
        sorted_wavfiles = [
            wavfile
            for idx, wavfile in sorted(
                [(int(fp.name.split('_')[-1].split('.')[0]), fp) for fp in list(steps_dir.glob('*.wav'))],
                key=lambda x: x[0],
            )
        ]
        steps = [librosa.load(file, sr=self._samp_freq)[0] for file in sorted_wavfiles]
        return steps


def load_continua(continua_df):
    continua_dict = {voice: {} for voice in continua_df['voice'].unique()}
    for endpointA, endpointB, tokenA, tokenB, voice, file_dir in continua_df.to_numpy():
        continua_dict[voice][f"{endpointA}-{endpointB}"] = Continuum(
            file_dir=ROOT / file_dir,
            endpoints=(endpointA, endpointB),
            tokens=(tokenA, tokenB),
            voice=voice,
            samp_freq=SAMP_FREQ,
        )
    return continua_dict
