import { spawn } from "node:child_process";
import { existsSync } from "node:fs";
import { join } from "node:path";

const host = "localhost";
const port = process.env.PORT || "3000";
const prettyUrl = `http://${host}:${port}`;

const npmCommand = process.platform === "win32" ? "npm.cmd" : "npm";
const nextBinary = join(process.cwd(), "frontend", "node_modules", ".bin", "next");

let announced = false;
let child;

const announce = () => {
  if (!announced) {
    announced = true;
    console.log(`Derm AI Nexus is running at ${prettyUrl}`);
  }
};

const wireDevOutput = (proc) => {
  proc.stdout.on("data", (chunk) => {
    const text = String(chunk);

    if (
      text.includes("Ready in") ||
      text.includes("Local:") ||
      text.includes("started server")
    ) {
      announce();
    }
  });

  proc.stderr.on("data", (chunk) => {
    const text = String(chunk);
    if (!announced && text.toLowerCase().includes("error")) {
      console.error(text.trim());
    }
  });

  proc.on("close", (code) => {
    if (!announced && code === 0) {
      announce();
    }
    process.exit(code ?? 0);
  });
};

const startDevServer = () => {
  child = spawn(`${npmCommand} run dev --prefix frontend -- -p ${port}`, {
    stdio: ["inherit", "pipe", "pipe"],
    env: process.env,
    shell: true,
  });
  wireDevOutput(child);
};

const ensureInstallAndRun = () => {
  if (existsSync(nextBinary)) {
    startDevServer();
    return;
  }

  console.log("First run setup: installing dependencies...");
  const install = spawn(`${npmCommand} install --prefix frontend`, {
    stdio: "inherit",
    env: process.env,
    shell: true,
  });

  install.on("close", (code) => {
    if (code !== 0) {
      process.exit(code ?? 1);
      return;
    }
    startDevServer();
  });
};

ensureInstallAndRun();

["SIGINT", "SIGTERM"].forEach((sig) => {
  process.on(sig, () => {
    if (child) {
      child.kill(sig);
    }
  });
});
