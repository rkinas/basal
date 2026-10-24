"""The Inspect view: basal-viz's Explainer for one downloaded model, loaded into basal-ui's own process (the page reads
every layer of the PyTorch model) and served by basal-viz's own app mounted at /viz (basal/viz.py make_app)."""
import gc
import threading
from types import SimpleNamespace

from basal.viz import Explainer

from .jobs import Cancelled

NOT_INSPECTABLE = ("basal-viz reads the bf16 PyTorch weights of basal-1.0, basal-1.5 and basal-2.0; download the bf16 "
                   "variant")


def inspectable(variant) -> str | None:
    """None when basal-viz can read the variant, else why not (MLX / GGUF / FP8 / NVFP4 ports, basal-omni)."""
    if variant.port != "bf16" or variant.experimental:
        return NOT_INSPECTABLE
    return None


def device(hw) -> str:
    return "mps" if hw.platform == "apple" else "cuda" if hw.platform == "cuda" else "cpu"


class Inspector:
    """At most one Explainer; its basal-serve batching loop (Server.worker) runs on basal-ui's event loop, which the
    page's /viz/api/explain awaits on."""

    def __init__(self):
        self.ex, self.model, self.state_name, self.error, self.calibration = None, None, "idle", None, None
        self.loop, self.worker, self.gen = None, None, 0
        self.lock = threading.RLock()
        self.worker_done = threading.Event()
        self.worker_done.set()

    def bind(self, loop):
        """The event loop of basal-ui's HTTP server (captured at start-up)."""
        self.loop = loop

    def explainer(self):
        with self.lock:
            if self.ex is None or self.state_name != "ready":
                raise RuntimeError("no model is being inspected: start one in basal-ui's Inspect view")
            return self.ex

    def start(self, job, variant, local_dir, hw, calibration=None, orders=None, on_exit=None):
        """Load the Explainer and run its batching loop on the UI event loop."""
        self.stop()
        with self.lock:
            if job.cancel.is_set():
                raise Cancelled()
            gen = self.gen
            self.model, self.state_name, self.error = variant.id, "loading", None
            self.calibration = calibration
        job.progress("loading model")
        if job.cancel.is_set():
            raise Cancelled()
        try:
            ex = Explainer(SimpleNamespace(model=str(local_dir), dtype="bfloat16", device=device(hw), orders=orders,
                                           calibration=calibration))
        except BaseException as exc:
            with self.lock:
                if gen == self.gen:
                    self.state_name, self.error = "error", str(exc)
            raise
        with self.lock:
            if gen != self.gen or job.cancel.is_set():
                stale = True
            else:
                stale = False
                ex.name = ex.info["model"] = variant.id  # a cache snapshot directory would show its commit hash
                self.ex = ex

                async def run_worker():
                    await ex.srv.worker()

                def finished(task):
                    if on_exit is not None:
                        on_exit()
                    self.worker_done.set()

                task = {}
                scheduled = threading.Event()

                def create_worker():
                    worker = self.loop.create_task(run_worker())
                    worker.add_done_callback(finished)
                    task["worker"] = worker
                    scheduled.set()

                self.worker_done.clear()
                self.loop.call_soon_threadsafe(create_worker)
                scheduled.wait()
                self.worker = task["worker"]
                self.state_name = "ready"
        if stale:
            del ex
            gc.collect()
            self._free_engine_memory()
            raise Cancelled()
        return {"model": variant.id, "info": ex.info}

    @staticmethod
    def _free_engine_memory():
        try:
            from .hardware import free_engine_memory
            free_engine_memory()
        except ImportError:
            pass

    def stop(self):
        with self.lock:
            self.gen += 1
            was_loading = self.state_name == "loading"
            worker, self.worker = self.worker, None
            self.ex, self.model, self.state_name, self.error, self.calibration = None, None, "idle", None, None
            if worker is not None:
                self.loop.call_soon_threadsafe(worker.cancel)
        exited = worker is None or self.worker_done.wait(timeout=5)
        if exited and not was_loading:
            gc.collect()
            self._free_engine_memory()


    def state(self) -> dict:
        """What the Playground's handoff compares with the served model: the engine, option orders and calibration."""
        with self.lock:
            out = {"model": self.model, "state": self.state_name, "error": self.error}
            if self.ex is not None and self.state_name == "ready":
                info = self.ex.info
                out.update(mode=info["mode"], device=info["device"], orders=info["orders"], calibration=self.calibration,
                           temperatures=info["temperatures"], temperature_per_bucket=info["temperature_per_bucket"])
            return out
