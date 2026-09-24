#!/usr/bin/env bash
# Replays the V27-T3 matrix with the staged hosts used on 2026-09-24.
# Pass a fresh JSON path as $1; benchmark_trio.py refuses to overwrite output.
set -euo pipefail

output_path="${1:-experiments/trio_v27/results_reproduction.json}"

python3 experiments/trio_v27/benchmark_trio.py \
  --worker 'B580|ssh|Windows 11@10.0.0.146|"C:\Users\WINDOW~1\sp_trio_v27_c084da1\qwen_vk.exe" "C:\Users\WINDOW~1\sp_trio_v27_c084da1\qwen05.sp" --serve' \
  --worker 'RX570|ssh|robson@10.0.0.253|/home/robson/sp_trio_v27_c084da1/qwen_vk /home/robson/sp_v23/qwen05.sp --serve' \
  --worker 'M4|local|"/Users/robson/Downloads/plano para o seedplane/artifacts/v22b-main-ci/seedplane-native-macOS-ARM64/qwen_vk" "/tmp/seedplane-trio-v27.MypJRc/qwen05.sp" --serve' \
  --identity 'B580|c0fcb93563c715947f54eb6f7b851106918187a3897985e0fe4429d8318c1e3a|214eb5b86779cadd28a1c6aa2d9d8574704f0b609423eb117cea04dc8fe0380c|ef9f3f59f925e46a303193d7b88a979a899c53b5fec2ed0ff679b61fdf3cbe49|c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539|8e342f5e8b8dab6a77886e395b9159b6e67fbd077344ef6a89b857642269e9b9' \
  --identity 'RX570|6b89b2e14e355186e04d3064da1909b5996b75bfef93c4d98c0212a04772be0b|214eb5b86779cadd28a1c6aa2d9d8574704f0b609423eb117cea04dc8fe0380c|ef9f3f59f925e46a303193d7b88a979a899c53b5fec2ed0ff679b61fdf3cbe49|c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539|8e342f5e8b8dab6a77886e395b9159b6e67fbd077344ef6a89b857642269e9b9' \
  --identity 'M4|f0bccaff7ea2e79ad11d17a56db4f178db67505dea8595b3fc3992bd35738aac|214eb5b86779cadd28a1c6aa2d9d8574704f0b609423eb117cea04dc8fe0380c|ef9f3f59f925e46a303193d7b88a979a899c53b5fec2ed0ff679b61fdf3cbe49|c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539|8e342f5e8b8dab6a77886e395b9159b6e67fbd077344ef6a89b857642269e9b9' \
  --network-note 'Mac controller/worker uses Wi-Fi en1, 10.0.0.92; avg RTT to B580 9.295 ms and X79 9.896 ms (5 pings each).' \
  --network-note 'B580 wired Realtek 1 Gbps; X79 enp7s0 100 Mb/s full duplex; iperf3 absent. Exploratory only, formal wired network gate unmet.' \
  --output "$output_path"
