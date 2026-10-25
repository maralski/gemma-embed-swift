#!/bin/bash
# Usage: scripts/make_speech_samples.sh [<out dir>]   (default models/audio/speech)
# Writes ten spoken sentences as NN.wav (16 kHz 16-bit mono) and NN.txt (the transcript) with macOS
# `say` and `afconvert`, for the cross-modal tests that check a clip ranks its own transcript first.
# The output is a synthetic macOS voice, so it lives under the gitignored models/ folder and must never
# be committed. The committed audio fixtures (scripts/make_fixture_audio.py) are synthetic signals with
# no voices. Running it again rewrites the same files. Then, with llama-server running as in
# scripts/oracle_audio.py:
#   python3 -I scripts/oracle_audio.py --speech models/audio/speech http://127.0.0.1:8094
set -euo pipefail
out="${1:-$(dirname "$0")/../models/audio/speech}"
voice="Samantha"
mkdir -p "$out"
sentences=(
  "Tomorrow will start cloudy with light showers, clearing to sunshine by the afternoon and a top of nineteen degrees."
  "Fry the onions slowly in butter until they turn golden, then stir in the garlic for one more minute."
  "The eight fifteen train to Riverside now departs from platform four and will stop at every station."
  "Water your tomato plants early in the morning, at the base of the stem, so the leaves stay dry."
  "Version two point four fixes the crash when exporting large files and adds a dark theme to the settings page."
  "The Harbour City Gulls won the final three goals to one, with two goals scored in the last ten minutes."
  "The new maritime museum opens on Saturday, with free entry for children and a gallery of old ship models."
  "To fix a flat bicycle tyre, remove the wheel, find the hole in the inner tube, and cover it with a patch."
  "The library will close early on Friday for maintenance, and books due that day can be returned on Monday."
  "Fold the beaten egg whites gently into the chocolate mixture, then pour it into the tin and bake for twenty minutes."
)
for i in "${!sentences[@]}"; do
  n=$(printf "%02d" "$i")
  say -v "$voice" -o "$out/$n.aiff" "${sentences[$i]}"
  afconvert -f WAVE -d LEI16@16000 -c 1 "$out/$n.aiff" "$out/$n.wav"
  rm "$out/$n.aiff"
  printf "%s\n" "${sentences[$i]}" > "$out/$n.txt"
  echo "$n.wav $n.txt"
done
