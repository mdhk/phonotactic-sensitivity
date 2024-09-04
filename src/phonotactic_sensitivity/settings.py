import json
from phonotactic_sensitivity.paths import ROOT

SAMP_FREQ = 16000
models = json.load(open(ROOT / 'models/models.json', 'r'))
voices = {
    'A': {
        'id': 'en-US-Standard-A',
        'gender': 'male',
    },
    'E': {'id': 'en-US-Standard-E', 'gender': 'female'},
}
seed = 11750232842146764334
