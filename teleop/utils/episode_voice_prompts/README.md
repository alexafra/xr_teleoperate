# Episode voice prompts

These three fixed prompts are pre-rendered so teleoperation never performs neural speech
synthesis or network access in the control path. `AsyncEpisodeVoiceNotifier` plays them on a
background worker with PipeWire's `pw-play` and falls back to `spd-say` when required.

They were generated with Piper 1.8.0 and the `en_GB-cori-high` UK English female voice:

- Piper: https://github.com/OHF-Voice/piper1-gpl
- Voice model: https://huggingface.co/rhasspy/piper-voices/tree/main/en/en_GB/cori/high
- Voice-model repository license: MIT
- Source recordings: LibriVox public-domain material, as documented by the model card

The committed files are 22.05 kHz, mono, 16-bit PCM WAVs. Regenerating them is an explicit
asset update; runtime operation does not require Piper or the downloaded model.
