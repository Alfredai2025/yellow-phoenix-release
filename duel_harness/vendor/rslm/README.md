# Vendored: RSLM reference codec (Google, Apache-2.0)

`cells_codec.py` is the official reference implementation of **Rotated Scaled
Lloyd-Max (RSLM) Vector Quantization for Approximate Nearest Neighbor
Search**, vendored UNMODIFIED from:

    https://github.com/google-research/google-research/tree/master/rslm

* Upstream license: **Apache License, Version 2.0** — see `LICENSE` in this
  directory (full text, verbatim from upstream).
* The upstream file's own header documents that it is "code cells of
  rslm.ipynb, imports trimmed to codec needs only; codec code unmodified".
  No notices were stripped or altered by this project; the file is used as
  distributed.
* It is the byte-exact opponent in this harness's duel: the published duel
  cells were measured against this code, so the vendored copy is what the
  expected outputs correspond to.

To verify the vendored copy against upstream:

    curl -L -o /tmp/rslm_upstream.ipynb \
      https://raw.githubusercontent.com/google-research/google-research/master/rslm/rslm.ipynb

and diff the codec code cells. To fetch the whole upstream tree instead of
using this vendor directory:

    git clone --depth 1 --filter=blob:none --sparse https://github.com/google-research/google-research.git
    cd google-research && git sparse-checkout set rslm

Citation (upstream README): "RSLM enables high-ratio compression of dense
floating-point embeddings down to 1.0–4.0 bits per dimension"; cite the
upstream repository/paper for the RSLM method itself. This project claims no
ownership of the vendored file.
