"""Interactive chat using the SeedPlane Qwen engine."""
import argparse, sys, time
from tokenizers import Tokenizer
from .qwen_engine import Qwen2Engine, chat_prompt


def main():
    p = argparse.ArgumentParser()
    p.add_argument('bundle'); p.add_argument('--device', default='cpu'); p.add_argument('-n', '--max-new-tokens', type=int, default=256)
    p.add_argument('--temperature', type=float, default=0.7); p.add_argument('--top-k', type=int, default=40); p.add_argument('--top-p', type=float, default=.9)
    a = p.parse_args(); tok = Tokenizer.from_file(f'{a.bundle}/tokenizer.json'); engine = Qwen2Engine(a.bundle, a.device)
    history = [{'role': 'system', 'content': 'You are a helpful assistant.'}]
    while True:
        try: question = input('\nVocê> ').strip()
        except (EOFError, KeyboardInterrupt): print(); return
        if not question: continue
        history.append({'role': 'user', 'content': question}); ids = tok.encode(chat_prompt(history)).ids
        answer_ids = []; t0 = time.perf_counter(); first = None; print('Qwen> ', end='', flush=True)
        for token in engine.generate(ids, a.max_new_tokens, a.temperature, a.top_k, a.top_p, 151645):
            if first is None: first = time.perf_counter()
            answer_ids.append(token); text = tok.decode(answer_ids, skip_special_tokens=True)
            print('\rQwen> ' + text, end='', flush=True)
        elapsed = time.perf_counter() - (first or t0); answer = tok.decode(answer_ids, skip_special_tokens=True)
        print(f'\n[{len(answer_ids) / max(elapsed, 1e-9):.1f} tok/s decode; TTFT {(first or t0)-t0:.3f}s]')
        history.append({'role': 'assistant', 'content': answer})


if __name__ == '__main__': main()
