"""basal: fast, calibrated typed decisions (choice / yes-no / score) for Polish and English."""
import os

__version__ = "2.4.1"
# TORCHINDUCTOR_CACHE_DIR as given before anything imported torch's compiler (importing transformers' models makes
# Inductor write its /tmp default there): engine.persist_compile_cache keeps a directory the user chose
INDUCTOR_CACHE_DIR_GIVEN = bool(os.environ.get("TORCHINDUCTOR_CACHE_DIR"))
