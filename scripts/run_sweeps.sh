#!/usr/bin/env bash
# Download FineWeb (sp1024), then train every run (GPU). SLOTS = how many runs share one GPU at a
# time (default 4; tokenizer_fix, whose runs all use the 50,257-token GPT-2 vocabulary and have the
# highest peak memory of any sweep, at most 2). Finished runs are skipped and failed runs are retried,
# so the script can be re-run to resume; after a killed sweep, add --reclaim to the nanolm.sweep call
# to clear the run claims it left behind.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/env.sh
python -m nanolm.prepare --golf-shards 1 --dest data/golf
# tokenize Tiny Shakespeare once per tokenizer before runs share the GPU, so no two runs build the
# same cache at the same time
L3=external/DeepDataMiningLearning/DeepDataMiningLearning/llm/nanolm/configs/laptop/L3_tokenizer.yaml
for t in char byte word:2048 word:8192 sp_train:bpe:512 sp_train:bpe:2048 sp_train:unigram:2048 gpt2; do
  python -m nanolm.prepare -c "$L3" data.tokenizer="$t" > /dev/null
done
SLOTS=${SLOTS:-4}
gpus() { python -c "print(','.join(['0'] * $1))"; }
SWEEPS="tokenizers tokenizer_fix recipe positions scaling swiglu"
for s in $SWEEPS; do
  n=$SLOTS; [ "$s" = tokenizer_fix ] && n=$(( SLOTS < 2 ? SLOTS : 2 ))
  python -m nanolm.sweep --sweep configs/sweeps/$s.yaml --gpus "$(gpus $n)"
done
for s in $SWEEPS; do
  python -m nanolm.analyze runs/sweeps/$s --metric val_bpb > runs/sweeps/$s/analyze_stdout.txt
done
