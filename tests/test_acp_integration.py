#!/usr/bin/env python3
"""ACP protocol integration test."""

import json
import subprocess
import sys
import time
import select

AGENT_CODE = '''
import asyncio, dataclasses, sys, logging

# Redirect any logging to stderr before importing march
for h in logging.getLogger().handlers[:]:
    if isinstance(h, logging.StreamHandler) and h.stream is sys.stdout:
        h.stream = sys.stderr

@dataclasses.dataclass
class FakeResponse:
    content: str = "Hello!"
    total_tokens: int = 42
    total_cost: float = 0.001
    tool_calls_made: int = 0

class FakeAgent:
    async def run(self, msg, session):
        return FakeResponse(content=f"You said: {msg}")

async def main():
    from march.channels.acp import ACPChannel
    ch = ACPChannel()
    await ch.start(FakeAgent())

asyncio.run(main())
'''


def main():
    proc = subprocess.Popen(
        [sys.executable, '-c', AGENT_CODE],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, bufsize=1,
    )

    def send(msg):
        proc.stdin.write(json.dumps(msg) + '\n')
        proc.stdin.flush()

    def read_lines(timeout=1.0):
        lines = []
        deadline = time.time() + timeout
        while time.time() < deadline:
            ready = select.select([proc.stdout], [], [], 0.05)[0]
            if ready:
                line = proc.stdout.readline().strip()
                if line:
                    try:
                        parsed = json.loads(line)
                        if isinstance(parsed, dict) and 'jsonrpc' in parsed:
                            lines.append(parsed)
                    except json.JSONDecodeError:
                        pass  # Skip non-JSON (log output)
            elif lines:
                break
        return lines

    passed = 0
    failed = 0

    def check(name, condition):
        nonlocal passed, failed
        if condition:
            print(f'  ✅ {name}')
            passed += 1
        else:
            print(f'  ❌ {name}')
            failed += 1

    # === Initialize ===
    print('Initialize:')
    send({'jsonrpc': '2.0', 'id': 1, 'method': 'initialize', 'params': {
        'protocolVersion': 1,
        'clientCapabilities': {'terminal': True, 'fs': {'readTextFile': True}},
        'clientInfo': {'name': 'test-ide', 'version': '1.0'},
    }})
    msgs = read_lines()
    check('got response', len(msgs) == 1)
    if msgs:
        r = msgs[0].get('result', {})
        check('protocolVersion=1', r.get('protocolVersion') == 1)
        check('agentInfo.name=march', r.get('agentInfo', {}).get('name') == 'march')
        check('has agentCapabilities', 'agentCapabilities' in r)
        check('authMethods=[]', r.get('authMethods') == [])

    # === Session New ===
    print('\nSession New:')
    send({'jsonrpc': '2.0', 'id': 2, 'method': 'session/new', 'params': {
        'workspacePath': '/tmp/test-project',
    }})
    msgs = read_lines()
    check('got response', len(msgs) == 1)
    session_id = ''
    if msgs:
        session_id = msgs[0].get('result', {}).get('sessionId', '')
        check('has sessionId', len(session_id) > 0)

    # === Prompt ===
    print('\nSession Prompt:')
    send({'jsonrpc': '2.0', 'id': 3, 'method': 'session/prompt', 'params': {
        'sessionId': session_id,
        'content': [{'type': 'text', 'text': 'hello world'}],
    }})
    msgs = read_lines(timeout=2.0)
    updates = [m for m in msgs if m.get('method') == 'session/update']
    responses = [m for m in msgs if m.get('id') == 3 and 'result' in m]
    check('got session/update', len(updates) >= 1)
    check('got prompt response', len(responses) == 1)
    if responses:
        check('stopReason=endTurn', responses[0]['result'].get('stopReason') == 'endTurn')
    if updates:
        text = updates[0].get('params', {}).get('update', {}).get('content', {}).get('text', '')
        check('response contains text', 'hello world' in text.lower() or len(text) > 0)

    # === Bad Session ===
    print('\nBad Session:')
    send({'jsonrpc': '2.0', 'id': 4, 'method': 'session/prompt', 'params': {
        'sessionId': 'nonexistent',
        'content': [{'type': 'text', 'text': 'hi'}],
    }})
    msgs = read_lines()
    check('got error', len(msgs) == 1 and 'error' in msgs[0])
    if msgs and 'error' in msgs[0]:
        check('error code -32602', msgs[0]['error']['code'] == -32602)

    # === Unknown Method ===
    print('\nUnknown Method:')
    send({'jsonrpc': '2.0', 'id': 5, 'method': 'foo/bar', 'params': {}})
    msgs = read_lines()
    check('got error', len(msgs) == 1 and 'error' in msgs[0])
    if msgs and 'error' in msgs[0]:
        check('error code -32601', msgs[0]['error']['code'] == -32601)

    # === Shutdown ===
    print('\nShutdown:')
    send({'jsonrpc': '2.0', 'id': 6, 'method': 'shutdown', 'params': {}})
    msgs = read_lines()
    check('got response', len(msgs) == 1)

    proc.terminate()

    print(f'\n{"=" * 40}')
    print(f'Results: {passed} passed, {failed} failed')
    if failed:
        sys.exit(1)
    else:
        print('🎉 All ACP protocol tests passed!')


if __name__ == '__main__':
    main()
