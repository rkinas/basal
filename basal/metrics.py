"""Prometheus text exposition for GET /metrics of basal-serve and basal-router. No torch here: basal-router imports
this module alone (importing basal.server for it took 6-9 s on the router's event loop at the first scrape)."""


def prometheus(metrics):
    """Prometheus text exposition of [(name, help, type, [(labels dict, value)])]; label values escaped as the format
    asks (backslash, newline, double quote)."""
    def esc(v):
        return str(v).replace("\\", "\\\\").replace("\n", "\\n").replace('"', '\\"')
    out = []
    for name, help_, kind, samples in metrics:
        out += [f"# HELP {name} " + str(help_).replace("\\", "\\\\").replace("\n", "\\n"), f"# TYPE {name} {kind}"]
        for labels, value in samples:
            lab = ",".join(f'{k}="{esc(v)}"' for k, v in labels.items())
            out.append(f"{name}{{{lab}}} {value}" if lab else f"{name} {value}")
    return "\n".join(out) + "\n"
