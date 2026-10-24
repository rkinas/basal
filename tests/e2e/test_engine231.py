"""2.3.1 release gate: real local CPU/MLX paths; browser error cases explicitly simulate child responses."""
import json
import time

import pytest

from tests.e2e.support import CHOICE
from tests.e2e.test_pages import browser, page, view, sync_api


def test_cpu_packed_can_serve_and_restart(ui, bf16_model):
    model_id = bf16_model['id']
    for model in ui.status()["models"]:
        ui.unload(model["id"])
    chosen = {'mode': 'mps', 'device': 'cpu', 'batches': 'default'}
    try:
        loaded = ui.load(model_id, chosen)
        assert loaded['options'] == chosen
        assert ui.decide(model_id, CHOICE)['answers']['dept']['choice']
        ui.post('/api/restart', {'id': model_id})
        loaded = ui.wait_model(model_id)
        assert loaded['state'] == 'ready' and loaded['options'] == chosen
        assert ui.decide(model_id, CHOICE)['answers']['dept']['choice']
    finally:
        ui.unload(model_id)


def test_mlx_packed_and_health(ui, text_model, hardware):
    if hardware['platform'] != 'apple' or 'mlx' not in hardware['engines']:
        pytest.skip('MLX acceptance needs Apple Silicon with MLX')
    model_id = text_model['id']
    eligibility = ui.get('/api/models/options', params={'id': model_id})
    if eligibility['mlx_packed_reason']:
        pytest.skip(eligibility['mlx_packed_reason'])
    for model in ui.status()["models"]:
        ui.unload(model["id"])
    try:
        loaded = ui.load(model_id, {'mlx_packed': True})
        assert '--mlx-packed' in loaded['command']
        assert ui.decide(model_id, CHOICE)['answers']['dept']['choice']
        ui.post('/api/restart', {'id': model_id})
        assert ui.wait_model(model_id)['options']['mlx_packed'] is True
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            health = ui.model(model_id)['health']
            if health == {'reachable': True, 'running': 0, 'queued': 0}:
                break
            time.sleep(.2)
        assert health == {'reachable': True, 'running': 0, 'queued': 0}
    finally:
        ui.unload(model_id)


def test_browser_errors_health_warnings_and_options(page, text_model):
    """Real Chromium and app; errors/counter edge cases/warnings below are simulated, not GPU OOM acceptance."""
    view(page, 'playground')
    page.check('#editJsonToggle')
    page.fill('#requestJson', json.dumps(CHOICE))
    for status, code, message in [(422, 'INVALID_REQUEST', 'Invalid request'), (503, 'OUT_OF_MEMORY', 'GPU memory'),
                                  (500, 'INTERNAL_ERROR', 'Internal server error')]:
        page.route('**/api/decide', lambda route, request, status=status, code=code: route.fulfill(
            status=status, content_type='application/json', headers={'x-request-id': 'browser-case'},
            body=json.dumps({'error': 'retained diagnostic', 'code': code})))
        page.click('#sendButton')
        sync_api.expect(page.locator('#decideError')).to_contain_text(f'HTTP {status}')
        sync_api.expect(page.locator('#decideError')).to_contain_text(message)
        sync_api.expect(page.locator('#decideError')).to_contain_text('retained diagnostic')
        sync_api.expect(page.locator('#decideError')).to_contain_text('browser-case')
        sync_api.expect(page.locator('#sendButton')).to_be_enabled()
        page.unroute('**/api/decide')
    page.evaluate('clearTimeout(pollTimer)')
    view(page, 'models')
    for health, message in [({'reachable': True, 'running': None, 'queued': None}, 'load counters unavailable'),
                             ({'reachable': False}, 'unreachable'),
                             ({'reachable': True, 'running': 0, 'queued': 0}, 'idle'),
                             ({'reachable': True, 'running': 2, 'queued': 1}, '2 running · 1 queued')]:
        page.evaluate('h => renderLoadedRows([{id:"simulated",state:"ready",health:h}])', health)
        sync_api.expect(page.locator('.model-load')).to_have_text(message)
    page.evaluate('''() => renderCalibrationState({running:false, phase:'done_with_warnings',
        result:{file:'test.json',warnings:['insufficient FIT labels','CERT does not confirm','0 families tested']}})''')
    sync_api.expect(page.locator('#calibrationProgress')).to_contain_text('Completed with warnings')
    sync_api.expect(page.locator('#calibrationProgress')).to_contain_text('0 families tested')
    sync_api.expect(page.locator('#runCalibration')).to_be_enabled()
    page.evaluate('id => openLoadOptions(id)', text_model['id'])
    eligible = page.evaluate("!optionsState.data.mlx_packed_reason && ['mlx','mlx-q8'].includes(selectedOptionMode())")
    if eligible:
        sync_api.expect(page.locator('#optPackedNote')).to_contain_text('512 tokens')
    sync_api.expect(page.locator('#loadOptions')).to_contain_text('longest cached prefix')
    sync_api.expect(page.locator('#loadOptions')).to_contain_text('GPU memory')
    if eligible:
        page.check('#opt-mlx_packed')
        assert page.evaluate('collectOptions().mlx_packed') is True
    else:
        sync_api.expect(page.locator('#opt-mlx_packed')).to_be_disabled()
    page.click('#optionsCancel')
