# Episode voice prompts

The initialization prompt, three recording lifecycle prompts, and every-tenth-episode totals
from 10 through 1,000 are pre-rendered so teleoperation never performs neural speech synthesis
or network access.
`AsyncEpisodeVoiceNotifier` plays them on a background worker with PipeWire's `pw-play`.
The initialization and lifecycle prompts fall back to `spd-say` when required; a missing or
out-of-range episode total is skipped so the counter never changes to the robotic fallback
voice.

They were generated with Piper 1.8.0 and the `en_GB-cori-high` UK English female voice:

- Piper: https://github.com/OHF-Voice/piper1-gpl
- Voice model: https://huggingface.co/rhasspy/piper-voices/tree/main/en/en_GB/cori/high
- Voice-model repository license: MIT
- Source recordings: LibriVox public-domain material, as documented by the model card

The committed files are 22.05 kHz, mono, 16-bit PCM WAVs. The milestone bank lives under
`milestones/` as `episodes_saved_NNNN.wav`; each file was synthesized as the complete phrase
“NN episodes saved.” so the voice retains natural phrasing instead of concatenating words.
`initialized_successfully.wav` is the complete phrase “Initialized successfully.” and plays
once after the episode writer and recording controller are ready.
Regenerating them is an explicit asset update; runtime operation does not require Piper or
the downloaded model.
