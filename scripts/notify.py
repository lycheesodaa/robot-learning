#!/usr/bin/env python3
"""Run a command, tee its output to a log, and email the outcome when it ends (completed, failed or interrupted).

Usage: scripts/notify.py --suite DIR [--log FILE] -- COMMAND...
Settings come from .env at the repository root: NOTIFY_TO, SMTP_HOST, SMTP_PORT,
SMTP_USER, SMTP_PASSWORD and optionally SMTP_FROM (defaults to SMTP_USER).
"""
import argparse
from collections import Counter
from email.message import EmailMessage
import json
from pathlib import Path
import signal
import smtplib
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = ('NOTIFY_TO', 'SMTP_HOST', 'SMTP_PORT', 'SMTP_USER', 'SMTP_PASSWORD')


def load_env(path):
    env = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith('#') and '=' in line:
            key, value = line.split('=', 1)
            env[key.strip()] = value.strip().strip('"\'')
    missing = [k for k in REQUIRED if not env.get(k)]
    if missing:
        raise SystemExit(f'{path} is missing {", ".join(missing)}')
    return env


def suite_summary(suite):
    if not suite or not (suite / 'results.json').exists():
        return 'No suite results found.'
    rows = json.loads((suite / 'results.json').read_text())
    status = json.loads((suite / 'status.json').read_text()).get('status') if (suite / 'status.json').exists() else 'unknown'
    terms = Counter(r.get('termination', r.get('status')) for r in rows)
    released = sum(bool(r.get('release_occurred')) for r in rows)
    success = sum(bool(r.get('task_success')) for r in rows)
    return '\n'.join([f'Suite status: {status}', f'Episodes recorded: {len(rows)}',
                      f'Terminations: {dict(terms)}', f'Released: {released}', f'Task success: {success}'])


def send(env, subject, body):
    msg = EmailMessage()
    msg['Subject'], msg['From'], msg['To'] = subject, env.get('SMTP_FROM') or env['SMTP_USER'], env['NOTIFY_TO']
    msg.set_content(body)
    port = int(env['SMTP_PORT'])
    with (smtplib.SMTP_SSL if port == 465 else smtplib.SMTP)(env['SMTP_HOST'], port, timeout=30) as smtp:
        if port != 465:
            smtp.starttls()
        smtp.login(env['SMTP_USER'], env['SMTP_PASSWORD'])
        smtp.send_message(msg)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--suite', type=Path, help='Suite output directory, summarized in the email')
    parser.add_argument('--log', type=Path, help='Default: <suite>.log, or notify.log')
    parser.add_argument('--env', type=Path, default=ROOT / '.env')
    parser.add_argument('--test', action='store_true', help='Send a test email and exit')
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    env = load_env(args.env)
    if args.test:
        send(env, f'[sim] test from {socket.gethostname()}', 'Notification settings work.')
        print(f'Sent test email to {env["NOTIFY_TO"]}')
        return
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command:
        parser.error('Give the command after --')
    log = args.log or (args.suite.with_suffix('.log') if args.suite else Path('notify.log'))
    start = time.time()
    with open(log, 'a') as out:
        proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        try:
            for line in proc.stdout:
                sys.stdout.write(line); out.write(line); out.flush()
            code = proc.wait()
            outcome = 'completed' if code == 0 else f'failed (exit {code})'
        except KeyboardInterrupt:
            proc.send_signal(signal.SIGINT)
            code = proc.wait()
            outcome = 'interrupted'
    hours = (time.time() - start) / 3600
    tail = ''.join(log.read_text().splitlines(True)[-40:])
    name = args.suite.name if args.suite else Path(command[-1]).name
    body = '\n\n'.join([f'Command: {" ".join(command)}', f'Host: {socket.gethostname()}',
                        f'Duration: {hours:.2f} h', suite_summary(args.suite), f'Log: {log.resolve()}',
                        f'Last log lines:\n{tail}'])
    try:
        send(env, f'[sim] {name} {outcome}', body)
        print(f'Notification sent to {env["NOTIFY_TO"]}')
    except Exception as exc:
        print(f'Notification failed: {exc}', file=sys.stderr)
    raise SystemExit(code)


if __name__ == '__main__':
    main()
