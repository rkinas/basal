/** What went wrong:
 *  - "validation" (422): the request is invalid and must be changed;
 *  - "unavailable" (503): the GPU ran out of memory, or the server is still starting; a retry later or on another server may succeed;
 *  - "server" (500 and other statuses; 429, 502 and 504, from a proxy or a router in front, are retryable);
 *  - "timeout" / "network": no answer, or the response broke off (the request may or may not have run). */
export type ErrorKind = "validation" | "unavailable" | "server" | "timeout" | "network";

/** Statuses a later retry may get past: too many requests, and a gateway (proxy, router) without an answer. */
const RETRYABLE = new Set([429, 502, 503, 504]);

export class BasalError extends Error {
  readonly kind: ErrorKind;
  /** The HTTP status; undefined for "timeout" and "network". */
  readonly status?: number;
  /** The server's error message (both error formats: {"error": "..."} and --error-format openai). */
  readonly detail: string;
  /** The type of an --error-format openai body ("validation_error" | "server_error"). */
  readonly type?: string;
  /** The raw response body. */
  readonly body?: string;

  constructor(kind: ErrorKind, detail: string, opts: { status?: number; type?: string; body?: string; cause?: unknown } = {}) {
    super(opts.status !== undefined ? `basal: HTTP ${opts.status}: ${detail}` : `basal: ${detail}`, { cause: opts.cause });
    this.name = "BasalError";
    this.kind = kind;
    this.detail = detail;
    this.status = opts.status;
    this.type = opts.type;
    this.body = opts.body;
  }

  /** True for the errors where the same request may succeed later (429, 502, 503, 504, timeout, network). */
  get retryable(): boolean {
    return this.kind === "unavailable" || this.kind === "timeout" || this.kind === "network" ||
      (this.status !== undefined && RETRYABLE.has(this.status));
  }
}

/** The error of a non-200 response. */
export function errorFromResponse(status: number, text: string): BasalError {
  let detail = text;
  let type: string | undefined;
  try {
    const j = JSON.parse(text) as { error?: unknown };
    if (typeof j.error === "string") detail = j.error;
    else if (j.error && typeof j.error === "object") {
      const e = j.error as { message?: unknown; type?: unknown };
      if (typeof e.message === "string") detail = e.message;
      if (typeof e.type === "string") type = e.type;
    }
  } catch {
    // not JSON: the text is the detail
  }
  const kind: ErrorKind = status === 422 ? "validation" : status === 503 ? "unavailable" : "server";
  return new BasalError(kind, detail, { status, type, body: text });
}
