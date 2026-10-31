"""
shrink_models.py - make Keras .h5 model files smaller WITHOUT changing predictions.

Keras saves the optimizer state (Adam keeps 2 extra values per weight) inside the .h5.
That is only needed to CONTINUE training, not to predict. This script re-saves every
.h5 file in a folder without it and checks that predictions are identical.

Usage (run in the SAME environment you trained in):
    python shrink_models.py  "C:/path/to/model/folder"
Originals are kept as  <name>.h5.bak
"""
import os, shutil, sys
from pathlib import Path
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import numpy as np
import tensorflow as tf

folder = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
for f in sorted(folder.glob("*.h5")):
    before = f.stat().st_size
    model = tf.keras.models.load_model(f, compile=False)
    shapes = [i.shape for i in model.inputs]
    x = [np.random.randint(0, 50, size=(4,) + tuple(d or 1 for d in s[1:])) for s in shapes]
    x = x[0] if len(x) == 1 else x
    ref = model.predict(x, verbose=0)

    backup = f.with_suffix(".h5.bak")
    shutil.copy2(f, backup)
    model.save(f, include_optimizer=False)

    new = tf.keras.models.load_model(f, compile=False).predict(x, verbose=0)
    as_list = lambda o: list(o.values()) if isinstance(o, dict) else (list(o) if isinstance(o, (list, tuple)) else [o])
    same = all(np.allclose(a, b, atol=1e-6) for a, b in zip(as_list(ref), as_list(new)))
    after = f.stat().st_size
    print(f"{f.name}: {before/1e6:.1f} MB -> {after/1e6:.1f} MB  "
          f"({100*(1-after/before):.0f}% smaller)  predictions identical: {same}")
    if not same:
        shutil.copy2(backup, f); print("   restored original (predictions changed)")
