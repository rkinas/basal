// Starts basal-serve on the repo's stub engine (test/stub_server.py) for a test file.
import { spawn } from "node:child_process";
import net from "node:net";
import path from "node:path";
import { fileURLToPath } from "node:url";

const here = path.dirname(fileURLToPath(import.meta.url));
const repo = path.resolve(here, "../../..");

/** The Python of an environment with the repo's dependencies (BASAL_PYTHON; default python3). */
const python = process.env.BASAL_PYTHON ?? "python3";

function freePort() {
  return new Promise((resolve, reject) => {
    const s = net.createServer();
    s.listen(0, "127.0.0.1", () => {
      const { port } = s.address();
      s.close(() => resolve(port));
    });
    s.on("error", reject);
  });
}

/** -> {url, stop()}; extra: arguments of stub_server.py (--error-format openai, --api-key K). */
export async function startServer(extra = []) {
  const port = await freePort();
  const p = spawn(python, [path.join(here, "stub_server.py"), "--port", String(port), ...extra], {
    env: { ...process.env, PYTHONPATH: repo },
    stdio: ["ignore", "pipe", "pipe"],
  });
  let log = "";
  p.stderr.on("data", (d) => (log += d));
  await new Promise((resolve, reject) => {
    p.stdout.on("data", (d) => String(d).includes("ready") && resolve());
    p.on("exit", (c) => reject(new Error(`stub_server exited with ${c}: ${log}`)));
    setTimeout(() => reject(new Error(`stub_server did not start in 60 s: ${log}`)), 60000).unref();
  });
  const url = `http://127.0.0.1:${port}`;
  for (let i = 0; ; i++) {  // "ready" is printed before uvicorn binds the port
    try {
      if ((await fetch(`${url}/health`)).ok) break;
    } catch {
      if (i > 300) throw new Error("stub_server does not answer");
    }
    await new Promise((r) => setTimeout(r, 100));
  }
  return { url, stop: () => p.kill() };
}
