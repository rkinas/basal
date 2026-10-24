"""Custom calibration from basal-ui: basal-calibrate fits a calibration file on the user's labelled requests
(docs/CALIBRATION.md, input format), in basal-ui's process. Decision questions only: against a temporary, uncalibrated
basal-serve of the model (basal-calibrate --url). With task questions (classify / verify / extract with gold), whose
thresholds are certified on the answers as served with the fitted temperatures: with the engine in basal-ui's process
(basal-calibrate --model --mode). Files fitted elsewhere can be imported. The files live in
settings.calibrations_dir(model); one per model can be selected (basal-serve --calibration)."""
import contextlib
import io
import json
import os
import shutil
import time
from pathlib import Path

from basal.server import MODES, calibration_file, calibration_mismatch, certified_table

from .hardware import free_engine_memory

READY_POLL_S = 0.5
TARGETS = ("0.01", "0.05")


class _Lines(io.TextIOBase):
    """basal-calibrate's stdout and stderr, line by line into the job's log; `lines` keeps them for the report."""

    def __init__(self, log):
        self.log, self.lines, self.buffer = log, [], ""

    def writable(self):
        return True

    def write(self, text):
        self.buffer += text
        *done, self.buffer = self.buffer.split("\n")
        for line in done:
            self.log.append(line)
            self.lines.append(line)
        return len(text)

    def flush(self):
        if self.buffer:
            self.write("\n")


def load(path) -> dict:
    """A calibration file's JSON; ValueError when it is not one (temperatures or certified thresholds expected)."""
    try:
        cal = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError) as exc:
        raise ValueError(f"cannot read {path}: {getattr(exc, 'strerror', None) or exc}") from None
    except ValueError:
        raise ValueError(f"{path} is not JSON") from None
    if not isinstance(cal, dict) or not any(k in cal for k in ("temperature_per_prim", "thresholds", "task_thresholds")):
        raise ValueError(f"{path} is not a basal calibration file (no temperature_per_prim or thresholds)")
    return cal


def report_warnings(report):
    """Keep engine warning text, including FIT/CERT summaries and strict-check omissions."""
    return list(dict.fromkeys(line.strip().removeprefix("warning: ") for line in report.splitlines()
                             if line.strip().startswith("warning:") or
                             (line.strip().startswith("--strict:") and "0 families tested" in line)))


def summary(path, local_dir=None) -> dict:
    """What the Calibration table shows of a calibration file; `warning` when it belongs to another model (as
    basal-serve warns at start-up)."""
    cal = json.loads(Path(path).read_text(encoding="utf-8"))
    report_path = Path(path).with_suffix(".report.txt")
    warnings = report_warnings(report_path.read_text(encoding="utf-8")) if report_path.exists() else []
    out = {"warnings": warnings, "fitted_on": cal.get("fitted_on"), "gates": cal.get("gates"), "thresholds": certified_table(cal),
           "task_thresholds": cal.get("task_thresholds") or {}, "temperature_per_prim": cal.get("temperature_per_prim"),
           "engine": cal.get("engine"), "has_report": Path(path).with_suffix(".report.txt").exists(),
           "imported": Path(path).name.startswith("imported-")}
    if local_dir:  # basal-serve's own check, with --model the local directory as basal-ui starts it
        out["warning"] = calibration_mismatch(cal, str(local_dir), Path(local_dir))
    return out


def shipped_file(local_dir, mode, preset=None) -> Path:
    """The calibration file basal-serve loads for the model in `mode` without --calibration."""
    kind, quant = MODES[mode][:2]
    return calibration_file(Path(local_dir), kind, None, quant, preset)


def listing(model_id, settings_mod, local_dir=None) -> list[dict]:
    """The calibration files of the model in basal-ui (fitted or imported), newest first; unreadable files are listed
    with the error."""
    files = []
    for path in sorted(settings_mod.calibrations_dir(model_id).glob("*.json"), key=lambda p: p.stat().st_mtime,
                       reverse=True):
        rows = path.with_suffix(".rows.jsonl")
        entry = {"file": str(path), "when": path.stat().st_mtime, "rows": str(rows) if rows.is_file() else None}
        try:
            entry.update(summary(path, local_dir))
        except (OSError, ValueError) as exc:
            entry["error"] = str(exc)
        files.append(entry)
    return files


def unique_path(directory, stem) -> Path:
    """An unused JSON path in directory, with a numeric suffix when stem already exists."""
    directory = Path(directory)
    candidate = directory / f"{stem}.json"
    index = 1
    while candidate.exists() or candidate.with_suffix(".report.txt").exists():
        candidate = directory / f"{stem}-{index}.json"
        index += 1
    return candidate


def import_file(model_id, settings_mod, source) -> Path:
    """Copy a calibration file fitted elsewhere into the model's calibrations (with its report when one sits next to
    it); ValueError when it is not a calibration file."""
    source = Path(source).expanduser()
    load(source)
    directory = settings_mod.calibrations_dir(model_id)
    directory.mkdir(parents=True, exist_ok=True)
    target = unique_path(directory, f"imported-{time.strftime('%Y%m%d-%H%M%S')}-{source.stem}")
    # Reserve both names exclusively so imports never replace a calibration or its report.
    while True:
        try:
            fd = os.open(target, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            os.close(fd)
            break
        except FileExistsError:
            target = unique_path(directory, target.stem)
    try:
        shutil.copyfile(source, target)
        report = source.with_suffix(".report.txt")
        if report.exists():
            report_target = target.with_suffix(".report.txt")
            fd = os.open(report_target, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            os.close(fd)
            try:
                shutil.copyfile(report, report_target)
            except BaseException:
                report_target.unlink(missing_ok=True)
                raise
    except BaseException:
        target.unlink(missing_ok=True)
        raise
    return target


def labelled(input_path):
    """(decision requests, labelled task decisions) of a basal-calibrate input; ValueError names the problem line."""
    from basal import calibrate
    try:
        bodies, _, _, units = calibrate.read(str(input_path))
    except calibrate.InputError as exc:
        raise ValueError(str(exc)) from None
    return bodies, units


def engine_flags(flags) -> list[str]:
    """The basal-serve flags basal-calibrate's engine takes (add_engine_args), of a flag list such as
    catalog.serve_args + options.args; basal-serve's own (thinking, --strict-calibration) and --model are dropped."""
    from basal import calibrate
    known = {s for a in calibrate.parser()._actions for s in a.option_strings}
    out, keep = [], False
    for token in flags:
        if token.startswith("--"):
            keep = token in known and token != "--model"
        if keep:
            out.append(token)
    return out


def _basal_calibrate(job, argv, writer):
    """basal-calibrate's main with its output in the job's log (the redirection is process-wide: one calibration at
    a time)."""
    from basal import calibrate
    try:
        with contextlib.redirect_stdout(writer), contextlib.redirect_stderr(writer):
            calibrate.main(argv)
    except SystemExit as exc:
        if exc.code not in (None, 0):
            job.check()  # cancelled: the engine was stopped under basal-calibrate
            raise RuntimeError(exc.code if isinstance(exc.code, str) else
                               writer.lines[-1] if writer.lines else f"basal-calibrate exited ({exc.code})") from None
    finally:
        writer.flush()


def run_calibration(job, fleet, assessment, local_dir, input_path, out_path, targets=TARGETS, options=None,
                    engine_args=None, advanced=None):
    """Run basal-calibrate, either against a private child or an in-process engine."""
    writer = _Lines(job.log)
    advanced = advanced or {}
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    rows_path = Path(out_path).with_suffix(".rows.jsonl")
    common = ["--model", str(local_dir), "--out", str(out_path), "--targets", ",".join(targets)]
    for key, flag in (("min_per_type", "--min-per-type"), ("min_blend", "--min-blend"),
                      ("min_family", "--min-family"), ("min_per_set", "--min-per-set"),
                      ("concurrency", "--concurrency")):
        if key in advanced:
            common += [flag, str(advanced[key])]
    reference = advanced.get("reference")
    if reference:
        common += ["--reference", str(reference)]
    compare = advanced.get("compare", "shipped")
    if compare == "none":
        common.append("--no-calibration")
    elif isinstance(compare, str) and compare != "shipped":
        common += ["--calibration", compare]
    if advanced.get("strict"):
        common.append("--strict")
    if advanced.get("save_rows"):
        common += ["--save-rows", str(rows_path)]
    if advanced.get("from_rows"):
        common.append("--from-rows")
    if advanced.get("report_only"):
        common.append("--report-only")
    in_process = engine_args is not None
    argv = [str(input_path), *common, *(engine_args or ())]
    if advanced.get("from_rows"):
        _basal_calibrate(job, argv, writer)
    elif in_process:
        try:
            job.progress("calibrating (engine in basal-ui's process)")
            _basal_calibrate(job, argv, writer)
        finally:
            if advanced.get("reserve", True):
                fleet.release("calibrate")
            free_engine_memory()
    else:
        job.progress("starting model")
        runner = fleet.start_temporary("calibrate", assessment, job.log, options, extra_args=["--no-calibration"])
        try:
            while (state := runner.status())["state"] != "ready":
                job.check()
                if state["state"] == "error":
                    raise RuntimeError(state["error"])
                if state["state"] == "idle":
                    raise RuntimeError("the temporary basal-serve was stopped")
                time.sleep(READY_POLL_S)
            job.progress("calibrating")
            _basal_calibrate(job, [str(input_path), "--url", runner.base_url, *common], writer)
        finally:
            fleet.stop_temporary("calibrate")
    job.check()
    report = "\n".join(writer.lines)
    if advanced.get("report_only"):
        return {"file": None, "report": report, "report_only": True}
    if not Path(out_path).exists():
        raise RuntimeError("basal-calibrate wrote no file")
    Path(out_path).with_suffix(".report.txt").write_text(report + "\n", encoding="utf-8")
    return {"file": str(out_path), "report": report, "engine_in_process": in_process, **summary(out_path)}
