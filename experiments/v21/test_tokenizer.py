"""V21 G1: native tokenizer vs HF tokenizers (encode ids, decode round trip), plus an NFC check on every code point.

    python test_tokenizer.py <qwen_vk.exe> <bundle.sp> <wikitext2_test.txt> <workdir>
"""
import json, random, subprocess, sys, unicodedata
from pathlib import Path
import numpy as np
from tokenizers import Tokenizer

exe, bundle, wiki, work = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]); work.mkdir(parents=True, exist_ok=True)
hf = Tokenizer.from_file(str(bundle / 'tokenizer.json'))
specials = [a['content'] for a in json.load(open(bundle / 'tokenizer.json', encoding='utf-8'))['added_tokens']]

stress = '\n'.join([
    'Olá, mundo! Ação, coração, pão e maçã — "aspas" e ‘curvas’. É ÉÉ é.',
    unicodedata.normalize('NFD', 'Não é fácil: ação, você, pôr, àquele, Ñandú, Å, ǅ, ﬁ ligature'),
    'e\u0301 a\u0308 o\u0302\u0323 q\u0307\u0323 Å (A+ring) \u212b (angstrom) \u2126 ohm \u0344',
    '中文测试，汉字與繁體。日本語のテキスト、カタカナ。한국어 텍스트 ' + unicodedata.normalize('NFD', '한국어 분해'),
    'Emoji: 😀👍🏽 👨‍👩‍👧‍👦 🇧🇷 ❤️ 1️⃣ ©®™',
    'def f(x):\n\tif x  >=  10:\n\t\treturn x ** 2  # comment\r\n    else:\r\n        pass\n\n\n   \n',
    'spaces:\u00a0nbsp\u2028line-sep\u2029para\u001cfs\u001dgs \u3000ideographic\u200bzwsp\u2009thin',
    "It's I'M we'LL they'Re you'VE he'D she'S ſ'ſ 'ſ x'ſ don't 's 'lls 'rea",
    'Numbers 12345 3.14159 ٣٤٥ ⅷ ½ ²³ 1,000,000 0x1F',
    'العربية مع التشكيل: مُحَمَّد  हिन्दी देवनागरी क्षत्रिय',
    'Ελληνικά ΣΊΣΥΦΟΣ İstanbul ǈ ẞ ß Kelvin K',
    ''.join(specials) + ' <|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n<|endoftext|><|im_end',
    '   leading and trailing   ', '\n\n\n', ' \t \n \t', 'a' * 300, '!!!???...,,,', ' ' * 17 + 'x', '\r\n\r\n x',
])
random.seed(21)
blocks = [(0x20, 0x7E), (0xA0, 0x24F), (0x300, 0x36F), (0x370, 0x3FF), (0x400, 0x4FF), (0x590, 0x6FF), (0x900, 0x97F),
          (0x1100, 0x11FF), (0x2000, 0x206F), (0x3000, 0x30FF), (0x4E00, 0x4FFF), (0xAC00, 0xAD00), (0x1F300, 0x1F64F), (0x1D400, 0x1D4FF)]
def rnd():
    return ''.join(chr(random.randint(*random.choice(blocks))) for _ in range(random.randint(1, 40)))
fuzz = '\n'.join(rnd() for _ in range(3000))
all_cp = ''.join(chr(c) for c in range(0x20, 0x30000) if not (0xD800 <= c < 0xE000) and unicodedata.category(chr(c)) != 'Cn')

def native(flag, text_or_ids, suffix):
    src = work / f'in_{suffix}'; dst = work / f'out_{suffix}'
    if isinstance(text_or_ids, str): src.write_bytes(text_or_ids.encode('utf-8'))
    else: np.asarray(text_or_ids, np.int32).tofile(src)
    r = subprocess.run([exe, str(bundle), flag, str(src), str(dst)], capture_output=True, text=True)
    if r.returncode: raise SystemExit(r.stderr)
    return dst.read_bytes()

res = {}
for name, text in (('wikitext', wiki.read_text(encoding='utf-8')), ('stress', stress), ('fuzz_extra', fuzz)):
    ref = hf.encode(text).ids
    got = np.frombuffer(native('--tokenize-file', text, name), np.int32).tolist()
    first = next((i for i, (a, b) in enumerate(zip(ref, got)) if a != b), None if len(ref) == len(got) else min(len(ref), len(got)))
    ctx = None
    if first is not None:
        ctx = {'hf': [hf.decode([t]) for t in ref[max(0, first - 3):first + 5]], 'native': [hf.decode([t]) for t in got[max(0, first - 3):first + 5]]}
    back = native('--detok-file', got, name + '_ids').decode('utf-8', errors='replace')
    res[name] = {'chars': len(text), 'hf_tokens': len(ref), 'native_tokens': len(got), 'identical': ref == got,
                 'first_mismatch': first, 'context': ctx, 'roundtrip_equals_nfc': back == unicodedata.normalize('NFC', text)}
nfc_native = native('--nfc-file', all_cp, 'allcp').decode('utf-8')
nfd = unicodedata.normalize('NFD', all_cp)
res['nfc_extra'] = {'code_points': len(all_cp), 'nfc_equal': nfc_native == unicodedata.normalize('NFC', all_cp),
                    'nfc_of_nfd_equal': native('--nfc-file', nfd, 'nfd').decode('utf-8') == unicodedata.normalize('NFC', nfd),
                    'python_unicode': unicodedata.unidata_version}
res['G1_pass'] = res['wikitext']['identical'] and res['stress']['identical'] and res['wikitext']['roundtrip_equals_nfc'] and res['stress']['roundtrip_equals_nfc']
print(json.dumps(res, indent=1, ensure_ascii=True))
