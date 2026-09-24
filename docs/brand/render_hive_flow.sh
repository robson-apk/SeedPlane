#!/usr/bin/env bash
set -euo pipefail

# Reproducible, code-native README animation: one queue dispatches independent
# requests to persistent local agents. Labels and caveats live in README.md.
output="$(cd "$(dirname "$0")" && pwd)/hive-pull-flow.gif"
filter="drawbox=x=54:y=210:w=220:h=100:color=0x15233a:t=fill,
drawbox=x=54:y=210:w=220:h=100:color=0x5d82a7:t=4,
drawbox=x=720:y=48:w=500:h=108:color=0x15233a:t=fill,
drawbox=x=720:y=48:w=500:h=108:color=0x4abcf1:t=4,
drawbox=x=720:y=206:w=500:h=108:color=0x15233a:t=fill,
drawbox=x=720:y=206:w=500:h=108:color=0xe6b566:t=4,
drawbox=x=720:y=364:w=500:h=108:color=0x15233a:t=fill,
drawbox=x=720:y=364:w=500:h=108:color=0x6ee7c4:t=4,
drawbox=x=428:y=102:w=4:h=316:color=0x40536d:t=fill,
drawbox=x=274:y=258:w=156:h=4:color=0x40536d:t=fill,
drawbox=x=430:y=100:w=290:h=4:color=0x40536d:t=fill,
drawbox=x=430:y=258:w=290:h=4:color=0x40536d:t=fill,
drawbox=x=430:y=416:w=290:h=4:color=0x40536d:t=fill,
drawbox=x=92:y=238:w=34:h=12:color=0x8ca1bd:t=fill,
drawbox=x=92:y=260:w=124:h=8:color=0x354761:t=fill,
drawbox=x=92:y=278:w=90:h=8:color=0x354761:t=fill,
drawbox=x=754:y=82:w=48:h=48:color=0x142d43:t=fill,
drawbox=x=766:y=94:w=24:h=24:color=0x4abcf1:t=fill,
drawbox=x=754:y=240:w=48:h=48:color=0x382b1d:t=fill,
drawbox=x=766:y=252:w=24:h=24:color=0xe6b566:t=fill,
drawbox=x=754:y=398:w=48:h=48:color=0x18372f:t=fill,
drawbox=x=766:y=410:w=24:h=24:color=0x6ee7c4:t=fill,
drawbox=x=828:y=82:w=320:h=8:color=0x52637a:t=fill,
drawbox=x=828:y=100:w=260:h=8:color=0x354761:t=fill,
drawbox=x=828:y=118:w=290:h=8:color=0x354761:t=fill,
drawbox=x=828:y=240:w=320:h=8:color=0x52637a:t=fill,
drawbox=x=828:y=258:w=260:h=8:color=0x354761:t=fill,
drawbox=x=828:y=276:w=290:h=8:color=0x354761:t=fill,
drawbox=x=828:y=398:w=320:h=8:color=0x52637a:t=fill,
drawbox=x=828:y=416:w=260:h=8:color=0x354761:t=fill,
drawbox=x=828:y=434:w=290:h=8:color=0x354761:t=fill,
drawbox=x='276+mod(t*150,140)':y=252:w=14:h=14:color=0x8ca1bd:t=fill,
drawbox=x='436+mod(t*165,260)':y=94:w=14:h=14:color=0x4abcf1:t=fill,
drawbox=x='436+mod(t*145+85,260)':y=252:w=14:h=14:color=0xe6b566:t=fill,
drawbox=x='436+mod(t*175+160,260)':y=410:w=14:h=14:color=0x6ee7c4:t=fill,
fps=15,scale=960:-1:flags=lanczos,split[a][b];[a]palettegen=stats_mode=diff[p];[b][p]paletteuse=dither=bayer"

ffmpeg -hide_banner -loglevel error -y \
  -f lavfi -i "color=c=0x0b1220:s=1280x520:r=20:d=4.8" \
  -filter_complex "$filter" -loop 0 "$output"
echo "$output"
