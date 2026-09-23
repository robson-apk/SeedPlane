# Contributing

Contributions that improve correctness, reproducibility, hardware coverage, or documentation are welcome.

1. Open an issue describing the change or experimental question.
2. Keep experimental pass/fail criteria in `PROTOCOL.md` before collecting results.
3. Install with `python -m pip install -e .` and run `python -m unittest discover -s tests -v`.
4. Do not commit model licenses you cannot redistribute, credentials, private datasets, or machine-specific paths.
5. Include raw measurements and enough hardware/software provenance to reproduce performance claims.

Security reports must follow [SECURITY.md](SECURITY.md), not a public pull request.
