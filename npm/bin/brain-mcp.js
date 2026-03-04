#!/usr/bin/env node

const { execSync, spawn } = require('child_process');
const { existsSync, mkdirSync } = require('fs');
const { join } = require('path');
const os = require('os');

const CONFIG_DIR = join(os.homedir(), '.config', 'brain-mcp');
const VENV_DIR = join(CONFIG_DIR, '.venv');
const VENV_BIN = process.platform === 'win32'
  ? join(VENV_DIR, 'Scripts')
  : join(VENV_DIR, 'bin');
const BRAIN_MCP = join(VENV_BIN, 'brain-mcp');

function getPython() {
  for (const cmd of ['python3', 'python']) {
    try {
      const version = execSync(`${cmd} --version 2>&1`, { encoding: 'utf8' }).trim();
      const match = version.match(/(\d+)\.(\d+)/);
      if (match && parseInt(match[1]) >= 3 && parseInt(match[2]) >= 11) {
        return cmd;
      }
    } catch {}
  }
  return null;
}

function ensureVenv() {
  if (existsSync(BRAIN_MCP)) return;

  const python = getPython();
  if (!python) {
    console.error('Python 3.11+ required but not found.');
    console.error('   Install: https://www.python.org/downloads/');
    process.exit(1);
  }

  console.log('Installing brain-mcp... (one-time setup, ~2 min)');
  mkdirSync(CONFIG_DIR, { recursive: true });

  try {
    execSync(`${python} -m venv "${VENV_DIR}"`, { stdio: 'inherit' });
    const pip = join(VENV_BIN, 'pip');
    execSync(`"${pip}" install --upgrade pip -q`, { stdio: 'inherit' });

    // Check if running from local repo (development mode)
    const repoRoot = join(__dirname, '..', '..');
    const localPyproject = join(repoRoot, 'pyproject.toml');
    if (existsSync(localPyproject)) {
      console.log('(installing from local source)');
      execSync(`"${pip}" install -e "${repoRoot}" -q`, { stdio: 'inherit' });
    } else {
      execSync(`"${pip}" install brain-mcp -q`, { stdio: 'inherit' });
    }
    console.log('brain-mcp installed successfully!\n');
  } catch (err) {
    console.error('Installation failed:', err.message);
    process.exit(1);
  }
}

// Main
ensureVenv();

const child = spawn(BRAIN_MCP, process.argv.slice(2), {
  stdio: 'inherit',
  env: process.env,
});

child.on('exit', (code) => process.exit(code || 0));
