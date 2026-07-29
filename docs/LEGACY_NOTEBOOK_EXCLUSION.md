# Why `Experimentation_STRIVE.ipynb` Is Not Included

The supplied notebook is an earlier HotpotQA/Wikipedia retrieval-agent prototype. It predates STRIVE's
paper protocol, uses the retired mixed `S/T/N/R/IG` formulation, and has no training loop despite being
described as training code. Most importantly, its configuration contained plaintext API credentials.

It is excluded to prevent accidental credential disclosure and to avoid implying that its old metrics or
HotpotQA results were used in the current MATH-500/OlympiadBench paper. It should be archived privately
only after replacing every credential with an environment-variable lookup and rotating the exposed keys.
