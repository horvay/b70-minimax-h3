#!/usr/bin/env bun
// Samples an sd-cli process: Intel xe compute-engine busy % (drm fdinfo, same source as nvtop),
// process CPU %, VRAM and RSS. Classifies each sample (GPU / CPU / BOTH / IDLE), tracks run
// lengths, writes a CSV and prints a summary line every SUMMARY_S seconds.
//
// Usage: bun gpu-cpu-monitor.ts [--pid N] [--csv path] [--interval-ms 250] [--summary-s 30] [--log run.log]
import { appendFileSync, existsSync, readdirSync, readFileSync, writeFileSync } from "node:fs";
import { execSync } from "node:child_process";

const args = new Map<string, string>();
for (let i = 2; i < process.argv.length; i += 2) args.set(process.argv[i].replace(/^--/, ""), process.argv[i + 1]);

const intervalMs = Number(args.get("interval-ms") ?? 250);
const summaryS = Number(args.get("summary-s") ?? 30);
const csvPath = args.get("csv") ?? "gpu-cpu-monitor.csv";
const logPath = args.get("log");
const GPU_BUSY = 50; // % compute-engine busy
const CPU_BUSY = 300; // % of one core

function findPid(): number | null {
  if (args.get("pid")) return Number(args.get("pid"));
  try {
    const out = execSync("pgrep -x sd-cli", { encoding: "utf8" }).trim().split("\n")[0];
    return out ? Number(out) : null;
  } catch {
    return null;
  }
}

const clk = Number(execSync("getconf CLK_TCK", { encoding: "utf8" }).trim());

type Sample = { gpuCycles: number; gpuTotal: number; cpuTicks: number; vramKiB: number; rssKiB: number; t: number };

function sample(pid: number): Sample | null {
  let gpuCycles = 0;
  let gpuTotal = 0;
  let vramKiB = 0;
  try {
    for (const fd of readdirSync(`/proc/${pid}/fdinfo`)) {
      let text: string;
      try {
        text = readFileSync(`/proc/${pid}/fdinfo/${fd}`, "utf8");
      } catch {
        continue;
      }
      if (!text.includes("drm-driver:\txe")) continue;
      const num = (key: string) => Number(text.match(new RegExp(`^${key}:\\s+(\\d+)`, "m"))?.[1] ?? 0);
      gpuCycles += num("drm-cycles-ccs");
      gpuTotal = Math.max(gpuTotal, num("drm-total-cycles-ccs"));
      vramKiB += num("drm-resident-vram0");
    }
    const stat = readFileSync(`/proc/${pid}/stat`, "utf8");
    const fields = stat.slice(stat.lastIndexOf(")") + 2).split(" ");
    const cpuTicks = Number(fields[11]) + Number(fields[12]);
    const rssKiB = Number(readFileSync(`/proc/${pid}/status`, "utf8").match(/^VmRSS:\s+(\d+)/m)?.[1] ?? 0);
    return { gpuCycles, gpuTotal, cpuTicks, vramKiB, rssKiB, t: performance.now() };
  } catch {
    return null;
  }
}

function stage(): string {
  if (!logPath || !existsSync(logPath)) return "";
  const text = readFileSync(logPath, "utf8").slice(-20000).replace(/\r/g, "\n");
  const lines = text.split("\n").filter((l) => /\[INFO|\[ERROR|\/\d+ - /.test(l));
  return (lines.at(-1) ?? "").replace(/\x1b\[K/g, "").trim().slice(0, 100);
}

type State = "GPU" | "CPU" | "BOTH" | "IDLE";
const runs: Record<State, number[]> = { GPU: [], CPU: [], BOTH: [], IDLE: [] };
let current: State | null = null;
let runStart = 0;
let transitions = 0;
let sums = { gpu: 0, cpu: 0, n: 0 };

let pid = findPid();
while (pid === null) {
  await Bun.sleep(1000);
  pid = findPid();
}
if (!existsSync(csvPath)) writeFileSync(csvPath, "time_s,gpu_busy_pct,cpu_pct,state,vram_gib,rss_gib\n");
console.log(`monitoring sd-cli pid ${pid} every ${intervalMs} ms -> ${csvPath}`);

const t0 = performance.now();
let prev = sample(pid);
let lastSummary = performance.now();

const fmt = (xs: number[]) => {
  if (!xs.length) return "0x";
  const total = xs.reduce((a, b) => a + b, 0);
  return `${xs.length}x avg ${(total / xs.length).toFixed(1)}s max ${Math.max(...xs).toFixed(1)}s total ${total.toFixed(0)}s`;
};

while (true) {
  await Bun.sleep(intervalMs);
  const cur = sample(pid);
  if (!cur || !prev) {
    if (current) runs[current].push((performance.now() - runStart) / 1000);
    console.log(`sd-cli ${pid} exited. GPU ${fmt(runs.GPU)} | CPU ${fmt(runs.CPU)} | BOTH ${fmt(runs.BOTH)} | IDLE ${fmt(runs.IDLE)} | transitions ${transitions}`);
    process.exit(0);
  }
  const dt = (cur.t - prev.t) / 1000;
  const gpu = cur.gpuTotal > prev.gpuTotal ? (100 * (cur.gpuCycles - prev.gpuCycles)) / (cur.gpuTotal - prev.gpuTotal) : 0;
  const cpu = (100 * (cur.cpuTicks - prev.cpuTicks)) / clk / dt;
  const state: State = gpu >= GPU_BUSY && cpu >= CPU_BUSY ? "BOTH" : gpu >= GPU_BUSY ? "GPU" : cpu >= CPU_BUSY ? "CPU" : "IDLE";
  const now = performance.now();
  if (state !== current) {
    if (current) {
      runs[current].push((now - runStart) / 1000);
      transitions++;
    }
    current = state;
    runStart = now;
  }
  sums = { gpu: sums.gpu + gpu, cpu: sums.cpu + cpu, n: sums.n + 1 };
  appendFileSync(
    csvPath,
    `${((now - t0) / 1000).toFixed(2)},${gpu.toFixed(1)},${cpu.toFixed(0)},${state},${(cur.vramKiB / 1048576).toFixed(2)},${(cur.rssKiB / 1048576).toFixed(2)}\n`,
  );
  prev = cur;
  if ((now - lastSummary) / 1000 >= summaryS) {
    console.log(
      `[${((now - t0) / 1000).toFixed(0)}s] avg gpu ${(sums.gpu / sums.n).toFixed(0)}% cpu ${(sums.cpu / sums.n).toFixed(0)}% | ` +
        `GPU ${fmt(runs.GPU)} | CPU ${fmt(runs.CPU)} | BOTH ${fmt(runs.BOTH)} | IDLE ${fmt(runs.IDLE)} | flips ${transitions} | ` +
        `vram ${(cur.vramKiB / 1048576).toFixed(1)}G | ${stage()}`,
    );
    for (const k of Object.keys(runs) as State[]) runs[k] = [];
    transitions = 0;
    sums = { gpu: 0, cpu: 0, n: 0 };
    lastSummary = now;
  }
}
