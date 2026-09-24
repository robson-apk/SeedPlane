"""Python front end for the native SeedPlane runtime (`native/vulkan_decode`, seedplane-bundle/2).

The engine runs as a subprocess speaking JSON lines (`qwen_vk <bundle> --serve`). Tokenization, sampling, the chat
template and the shard-window KV state all live in the engine; this module only sends requests and streams text.

    from seedplane.native import NativeEngine
    with NativeEngine('./qwen05-native.sp') as eng:
        for piece in eng.chat('Olá! Quem é você?'):
            print(piece, end='', flush=True)
        print(eng.last)          # {'reason': 'stop', 'generated': ..., 'decode_tok_s': ..., ...}
"""
import json, os, shutil, subprocess, sys
from pathlib import Path

REPO_ENGINE = Path(__file__).resolve().parents[1] / 'native' / 'vulkan_decode' / 'build' / ('qwen_vk.exe' if os.name == 'nt' else 'qwen_vk')


def find_engine(explicit=None):
    """Engine binary: explicit path, $SEEDPLANE_NATIVE_ENGINE, the in-repo build, or `qwen_vk` on PATH."""
    for cand in (explicit, os.environ.get('SEEDPLANE_NATIVE_ENGINE'), REPO_ENGINE, shutil.which('qwen_vk')):
        if cand and Path(cand).is_file(): return str(cand)
    raise FileNotFoundError('native engine not found: build native/vulkan_decode (build.bat or CMake) or set SEEDPLANE_NATIVE_ENGINE')


class NativeEngine:
    def __init__(self, bundle, engine=None, mode=None, full=False, ctx=None, temperature=None, top_k=None, top_p=None, seed=None,
                 max_new_tokens=None, system=None, extra_args=()):
        args = [find_engine(engine), str(bundle), '--serve']
        for flag, val in (('--mode', mode), ('--ctx', ctx), ('--temperature', temperature), ('--top-k', top_k), ('--top-p', top_p),
                          ('--seed', seed), ('-n', max_new_tokens), ('--system', system)):
            if val is not None: args += [flag, str(val)]
        if full: args.append('--full')
        args += list(extra_args)
        self.proc = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.info = self._read(); self.last = None; self.last_tokens = []
        if 'ready' not in self.info: raise RuntimeError(f'engine failed to start: {self.info}')

    def _read(self):
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError('engine exited: ' + self.proc.stderr.read().decode('utf-8', 'replace').strip())
        return json.loads(line.decode('utf-8'))

    def _send(self, req):
        self.proc.stdin.write((json.dumps(req, ensure_ascii=False) + '\n').encode('utf-8')); self.proc.stdin.flush()

    def _call(self, req):
        self._send(req); r = self._read()
        if 'error' in r: raise RuntimeError(r['error'])
        return r

    def _stream(self, req):
        self.last_tokens = []
        self.last = None
        self._send(req)
        while True:
            r = self._read()
            if 'error' in r: raise RuntimeError(r['error'])
            if r.get('done'): self.last = r; return
            if 'token' in r: self.last_tokens.append(int(r['token']))
            if r['text']: yield r['text']

    def chat(self, content, system=None, reset=False, **sampling):
        """One user turn; yields the assistant's text as it is generated. The conversation stays in the engine."""
        req = {'op': 'chat', 'content': content, 'reset': reset, **sampling}
        if system is not None: req['system'] = system
        yield from self._stream(req)

    def generate(self, text=None, ids=None, reset=True, **sampling):
        """Raw continuation of `text` (special tokens allowed) or of token `ids`."""
        req = {'op': 'generate', 'reset': reset, **sampling}
        if ids is not None: req['ids'] = list(ids)
        else: req['text'] = text or ''
        yield from self._stream(req)

    def tokenize(self, text, special=True): return self._call({'op': 'tokenize', 'text': text, 'special': special})['ids']
    def detokenize(self, ids, skip_special=False): return self._call({'op': 'detokenize', 'ids': list(ids), 'skip_special': skip_special})['text']
    def state(self): return self._call({'op': 'state'})
    def reset(self): self._call({'op': 'reset'})

    def close(self):
        if self.proc.poll() is None:
            try: self.proc.stdin.close(); self.proc.wait(timeout=10)
            except Exception: self.proc.kill(); self.proc.wait()
        for f in (self.proc.stdin, self.proc.stdout, self.proc.stderr):
            try: f.close()
            except Exception: pass

    def __enter__(self): return self
    def __exit__(self, *exc): self.close()


def chat_repl(eng, out=sys.stdout):
    """Terminal chat loop over a NativeEngine (used by `seedplane chat --native`)."""
    i = eng.info
    print(f"SeedPlane native | {i['device']} | mode {i['mode']} | plan S={i['plan']['shard']} H={i['plan']['halo']} "
          f"K={i['plan']['sinks']} | /reset, /exit", file=out)
    while True:
        try: q = input('\nVocê> ').strip()
        except (EOFError, KeyboardInterrupt): print(file=out); return
        if q == '/exit': return
        if q == '/reset': eng.reset(); print('(nova conversa)', file=out); continue
        if not q: continue
        print('Qwen> ', end='', flush=True, file=out)
        for piece in eng.chat(q): print(piece, end='', flush=True, file=out)
        s = eng.last
        print(f"\n[{s['generated']} tokens, {s['decode_tok_s']:.1f} tok/s, prompt {s['prompt_tokens']} tokens in "
              f"{s['prefill_s']:.2f}s, position {s['position']}]", file=out)
