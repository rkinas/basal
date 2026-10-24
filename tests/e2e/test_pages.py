"""The pages of basal-ui in Chromium (Playwright, basal[e2e]): every view renders without a script error, the
Playground builds, sends and renders every question type, the load options dialog, Batch, History, Logs and Inspect."""
import json
import re

import pytest

from tests.e2e.support import CHOICE, EXAMPLES, advanced_requests

sync_api = pytest.importorskip("playwright.sync_api", reason="install basal[e2e] and run: playwright install chromium")

VIEWS = {"overview": "Overview", "models": "Models", "manager": "Model manager", "playground": "Playground",
         "batch": "Batch", "inspect": "Inspect", "api": "API server", "history": "History", "logs": "Logs"}
WAIT = 120_000  # ms: a decision of a large model, a model start


@pytest.fixture(scope="module")
def browser():
    with sync_api.sync_playwright() as p:
        b = p.chromium.launch()
        yield b
        b.close()


@pytest.fixture
def page(browser, ui, text_model):
    """A fresh tab on basal-ui with the text model loaded; fails the test on any uncaught script error."""
    ui.load(text_model["id"])
    context = browser.new_context(viewport={"width": 1440, "height": 1000})
    tab = context.new_page()
    errors = []
    tab.on("pageerror", lambda e: errors.append(f"pageerror: {e}"))
    tab.on("console", lambda m: errors.append(f"console: {m.text}") if m.type == "error"
           and "Failed to load resource" not in m.text else None)  # 4xx answers the page expects are not errors
    tab.set_default_timeout(WAIT)
    tab.goto(ui.url + "/#overview")
    tab.wait_for_function("() => typeof statusCache !== 'undefined' && statusCache && modelData.length > 0")
    yield tab
    context.close()
    assert not errors, errors


def view(page, name):
    page.click(f'.side-nav [data-view="{name}"]')
    sync_api.expect(page.locator("#pageTitle")).to_have_text(VIEWS[name])


def test_every_view_renders(page, text_model):
    for name in VIEWS:
        view(page, name)
        sync_api.expect(page.locator(f'section.view[data-page="{name}"]').first).to_be_visible()
    view(page, "overview")
    sync_api.expect(page.locator("#loadedModels")).to_contain_text(text_model["id"])
    sync_api.expect(page.locator("#footerState")).to_contain_text("ready")


@pytest.mark.parametrize("request_", advanced_requests(), ids=lambda r: r["id"])
def test_the_playground_builds_sends_and_renders_each_question_type(page, text_model, request_):
    view(page, "playground")
    page.select_option("#playgroundModel", text_model["id"])
    index = page.evaluate("id => examples.findIndex(e => e.file === 'advanced' && e.id === id)", request_["id"])
    page.select_option("#examples", str(index))
    built = page.evaluate("() => buildRequest()")
    assert built["state"] == request_["state"]
    assert built["questions"] == request_["questions"]  # exactly the request: the builder never rewrites it
    page.click("#sendButton")
    page.wait_for_selector("#results article.answer")
    sync_api.expect(page.locator("#decideError")).to_be_hidden()
    articles = page.locator("#results article.answer")
    sync_api.expect(articles).to_have_count(len(request_["questions"]))
    q_type = next(iter(request_["questions"].values()))["type"]
    sync_api.expect(articles.first.locator("h3 .chip").first).to_have_text(q_type)
    if q_type in ("compress", "extract"):
        sync_api.expect(page.locator("#results .source-view mark").first).to_be_visible()
    if q_type == "verify":
        sync_api.expect(articles.first.locator("tbody tr")).to_have_count(3)


RUBRIC = {"state": "Od wczoraj nie mogę zalogować się do aplikacji.",
          "questions": {"category": {"type": "classify", "option_keys": "show",
                                     "labels": [{"name": "bug", "rubric": "błąd aplikacji"}, {"name": "billing"}]},
                        "dept": {"type": "choice", "instructions": "Który dział?", "criteria": ["it", "sales"]}}}


def test_the_builder_keeps_what_it_cannot_write_in_its_own_form(page):
    view(page, "playground")
    page.evaluate("r => setRequest(r)", RUBRIC)
    assert page.evaluate("() => buildRequest().questions") == RUBRIC["questions"]
    cards = page.locator(".question")
    cards.nth(0).locator(".q-instructions-input").fill("Jaka to kategoria?")  # an edit elsewhere on the card
    cards.nth(1).locator(".q-instructions-input").fill("Do którego działu?")
    built = page.evaluate("() => buildRequest().questions")
    assert built["category"]["labels"] == RUBRIC["questions"]["category"]["labels"]  # still {name, rubric} objects
    assert built["category"]["option_keys"] == "show" and built["category"]["instructions"] == "Jaka to kategoria?"
    assert built["dept"]["criteria"] == ["it", "sales"] and "option_keys" not in built["dept"]
    cards.nth(0).locator('[data-f="labels"]').fill("bug: błąd aplikacji\nbilling: płatności\nother")
    assert page.evaluate("() => buildRequest().questions.category.labels") == [
        {"name": "bug", "rubric": "błąd aplikacji"}, {"name": "billing", "rubric": "płatności"}, {"name": "other"}]


def test_no_automation_claim_where_the_certificate_does_not_hold(page, text_model):
    """A media answer (calibration fitted on text) or one with unvalidated calibration is never "Automatable"."""
    view(page, "playground")
    probe = """([model, calibration]) => {
      const m = statusCache.models.find(x => x.id === model);
      const saved = m.entry; m.entry = {...saved, certified: {"0.01": {choice: 0.5}, "0.05": {choice: 0.4}},
                                        temperature_per_prim: {choice: 1.2}};
      renderResults({answers: {q: {type: 'choice', choice: 'a', probabilities: {a: 0.99, b: 0.01}, confidence: 0.99,
                                   ...(calibration ? {calibration} : {})}}},
                    {model, state: 's', questions: {q: {type: 'choice', criteria: {a: 'A', b: 'B'}}}});
      m.entry = saved;
      return document.querySelector('#results .chips.safety').textContent;
    }"""
    for calibration in ("text", "unvalidated"):
        strip = page.evaluate(probe, [text_model["id"], calibration])
        assert "Automatable" not in strip and "Not certified for this answer" in strip
    probe = """([model, answer, warning]) => {
      const m=statusCache.models.find(x=>x.id===model), saved=m.entry;
      m.entry={...saved,certified:{"0.01":{choice:0.5},"0.05":{choice:0.4}},temperature_per_prim:{choice:1.2},
               ...(warning?{calibration_warning:warning}:{})};
      renderResults({answers:{q:{type:'choice',choice:'a',probabilities:{a:0.99,b:0.01},confidence:0.99,...answer}}},
        {model,state:'s',questions:{q:{type:'choice',criteria:{a:'A',b:'B'}}}});
      m.entry=saved; return document.querySelector('#results .chips.safety').textContent;
    }"""
    for answer, warning in (({"deferred": True}, None), ({"max_error": "0.01"}, None), ({}, "missing calibration")):
        strip = page.evaluate(probe, [text_model["id"], answer, warning])
        assert "Automatable" not in strip and "review" in strip


def test_string_state_is_preserved_until_the_state_field_is_edited(page):
    view(page, "playground")
    literal = '{"not": "json"}'
    page.evaluate("state => setRequest({state, questions: {}})", literal)
    assert page.evaluate("() => buildRequest().state") == literal
    page.evaluate("() => setRequest({state: {structured: true}, questions: {}})")
    assert page.evaluate("() => buildRequest().state") == {"structured": True}
    page.evaluate("state => setRequest({state, questions: {}})", literal)
    page.fill("#stateInput", '{"edited": true}')
    assert page.evaluate("() => buildRequest().state") == {"edited": True}


def test_edit_json_rebuilds_the_builder_and_keeps_invalid_json_open(page):
    view(page, "playground")
    page.check("#editJsonToggle")
    page.fill("#requestJson", "{ invalid")
    page.click("#editJsonToggle")  # stays checked: the toggle refuses to leave invalid JSON
    sync_api.expect(page.locator("#editJsonToggle")).to_be_checked()
    sync_api.expect(page.locator("#editJsonArea")).to_be_visible()
    sync_api.expect(page.locator("#mediaError")).to_contain_text("Invalid request JSON")
    request = {"state": "builder from edited JSON", "questions": {"priority": {
        "type": "choice", "criteria": ["low", "high"]}}, "facts": "auto", "lang": "pl"}
    page.fill("#requestJson", json.dumps(request))
    page.uncheck("#editJsonToggle")
    sync_api.expect(page.locator("#builder")).to_be_visible()
    assert page.evaluate("() => buildRequest()")["questions"] == request["questions"]
    assert page.evaluate("() => buildRequest().state") == request["state"]


def test_the_builder_switches_types_and_edits_a_cost_matrix(page):
    view(page, "playground")
    page.evaluate("() => setRequest({state: 'Paczka nie dotarła.', questions: {}})")
    card = page.locator(".question").first
    card.locator(".q-type").select_option("act")
    card.locator('[data-f="criteria"]').fill("late: opóźnienie\nlost: zaginięcie")
    card.locator('[data-cost-mode="matrix"]').click()
    sync_api.expect(card.locator(".cost-table thead th")).to_have_text(["Action", "late", "lost", ""])
    card.locator(".add-action").click()
    row = card.locator(".cost-table tbody tr").last
    row.locator(".cost-action").fill("human")
    for cell in row.locator(".cost-cell").all():
        cell.fill("2")
    costs = page.evaluate("() => buildRequest().questions.question.costs")
    assert costs["human"] == {"late": 2, "lost": 2} and costs["defer"] == {"late": 1, "lost": 1}
    cell = row.locator(".cost-cell").first
    cell.fill("")
    assert page.evaluate("() => document.querySelector('.question')._error") == "every cost cell needs a number"
    costs = page.evaluate("() => buildRequest().questions.question.costs")
    assert "late" not in costs["human"]
    sync_api.expect(card.locator(".q-error")).to_contain_text("every cost cell needs a number")
    card.locator(".q-type").select_option("tool")
    card.locator('[data-f="tools"]').fill("not json")
    page.click("#sendButton")
    sync_api.expect(page.locator("#decideError")).to_contain_text("tools must be a JSON array")


def test_safety_strip_and_inspect_button_on_a_decision(page, text_model):
    view(page, "playground")
    page.evaluate("r => setRequest(r)", CHOICE)
    page.click("#sendButton")
    strip = page.locator("#results .chips.safety")
    sync_api.expect(strip.first).to_contain_text(
        re.compile("Automatable|Below the certified|No certificate|uncalibrated|calibrated"))
    sync_api.expect(page.locator("#inspectDecision")).to_be_visible()
    sync_api.expect(page.locator("#results")).to_contain_text("Longest prompt")


def test_the_load_options_dialog_lists_the_modes(page, text_model):
    view(page, "overview")  # the loaded model's ⚙: restart with other options
    page.click(f'#loadedModels .options-model[data-id="{text_model["id"]}"]')
    dialog = page.locator("#loadOptions")
    sync_api.expect(dialog).to_be_visible()
    sync_api.expect(page.locator("#optionsApply")).to_have_text("Restart")
    sync_api.expect(dialog.locator(f'input[name="optMode"][value="{text_model["mode"]}"]')).to_be_checked()
    assert dialog.locator('input[name="optMode"]').count() == 16
    assert dialog.locator('input[name="optMode"][disabled]').count() >= 1
    sync_api.expect(dialog.locator("#optExperimentalNote")).not_to_be_empty()
    for key, modes in (("gemma_attn", "fast"), ("eager_attn", "eager"), ("merge", "fast"), ("ple_offload", "eager")):
        field = dialog.locator(f'[data-opt="{key}"]')  # each switch shows for the backend that reads it
        assert modes in field.get_attribute("data-modes").split() and ("eager" in field.get_attribute(
            "data-modes").split()) == (modes == "eager")
        if text_model["mode"] not in field.get_attribute("data-modes").split():
            sync_api.expect(field).to_be_hidden()
    page.click("#optionsCancel")
    sync_api.expect(dialog).to_be_hidden()


def test_batch_view_runs_a_file_and_lists_the_answers(page, text_model, tmp_path):
    view(page, "batch")
    page.select_option("#batchModel", text_model["id"])
    page.fill("#batchInput", str(EXAMPLES / "questions.jsonl"))
    page.fill("#batchOutput", str(tmp_path / "answers.jsonl"))
    page.fill("#batchConcurrency", "4")
    page.click("#runBatch")
    sync_api.expect(page.locator("#downloadBatch")).to_be_visible()
    sync_api.expect(page.locator("#batchRows tbody tr")).to_have_count(20)
    sync_api.expect(page.locator("#batchSummary")).to_contain_text("20 items")


def test_history_and_logs_views(page, ui, text_model):
    ui.decide(text_model["id"], CHOICE)
    view(page, "history")
    rows = page.locator("#historyTable tr.history-row")
    sync_api.expect(rows.first).to_be_visible()
    rows.first.click()
    sync_api.expect(page.locator("#historyTable .history-detail .history-request")).to_contain_text('"questions"')
    view(page, "logs")
    sync_api.expect(page.locator("#logSources")).to_contain_text(text_model["id"].split("/")[-1])


def test_inspect_view_embeds_basal_viz_and_explains_a_playground_decision(page, ui, bf16_model):
    view(page, "inspect")
    page.select_option("#inspectModel", bf16_model["id"])
    page.click("#startInspect")
    frame_el = page.locator("#vizFrame")
    sync_api.expect(frame_el).to_be_visible(timeout=600_000)
    frame = page.frame_locator("#vizFrame")
    sync_api.expect(frame.locator("#modelchip")).to_contain_text(bf16_model["id"])
    view(page, "playground")
    request = {**CHOICE, "state": "Umowa podpisana 3 marca 2025.", "facts": "auto", "lang": "pl"}
    page.evaluate("r => setRequest(r)", request)
    page.click("#sendButton")
    page.click("#inspectDecision")
    sync_api.expect(page.locator("#pageTitle")).to_have_text("Inspect")
    sync_api.expect(frame.locator("#out .result").first).to_be_visible()
    sync_api.expect(frame.locator("#out")).to_contain_text("Do którego działu")
    sync_api.expect(frame.locator("#out .result").first).to_contain_text("Polish prompt")
    sync_api.expect(frame.locator("#out .result .prompt")).to_contain_text("Fakty pomocnicze")
    notice = page.locator("#inspectNotice")
    sync_api.expect(notice).to_be_visible()
    if page.input_value("#playgroundModel") != bf16_model["id"]:  # the served decision came from other weights
        sync_api.expect(notice).to_contain_text("weights: served")
    page.click("#stopInspect")
    sync_api.expect(frame_el).to_be_hidden()
    assert ui.get("/api/inspect")["state"] == "idle"


def test_the_inspect_handoff_names_every_difference_and_refuses_what_it_cannot_explain(page, text_model):
    """inspectHandoff on a served model whose engine, option orders and temperatures differ from the inspector's."""
    probe = """([model, request]) => {
      const saved = statusCache, served = statusCache.models.find(x => x.id === model);
      statusCache = {...saved, inspect: {model, state: 'ready', mode: 'eager', device: 'mps', orders: 1,
                                         temperatures: {choice: 9}},
                     models: [{...served, mode: 'mlx', options: {think: true},
                               entry: {...served.entry, orders: 2, temperature_per_prim: {choice: 1.2}}}]};
      try { return inspectHandoff({model, ...request}); } finally { statusCache = saved; }
    }"""
    mixed = {"state": "s", "think": "auto", "early_exit": "0.99", "facts": "auto", "questions": {
        "dept": {"type": "choice", "criteria": {"a": "A", "b": "B"}},
        "intent": {"type": "classify", "labels": {"x": "X", "y": "Y"}}}}
    handoff = page.evaluate(probe, [text_model["id"], mixed])
    assert list(handoff["body"]["questions"]) == ["dept"] and handoff["body"]["facts"] == "auto"
    assert not {"model", "think", "early_exit"} & set(handoff["body"])
    notes = " | ".join(handoff["notes"])
    for part in ("intent: task questions are left out", "engine: served mlx, explained eager on mps",
                 "option orders: served 2, explained 1", "calibration: the served temperatures differ",
                 "thinking: explained without the reasoning", "early exit 0.99"):
        assert part in notes, notes
    assert "media" in page.evaluate(probe, [text_model["id"], {**mixed, "media": [{"type": "image"}]}])["blocked"]
    only_tasks = {"state": "s", "questions": {"intent": mixed["questions"]["intent"]}}
    assert "only task questions" in page.evaluate(probe, [text_model["id"], only_tasks])["blocked"]
    other = page.evaluate("""([model, request]) => {
      const saved = statusCache; statusCache = {...saved, inspect: {model: 'Remek/basal-1.0-max', state: 'ready'}};
      try { return inspectHandoff({model, ...request}); } finally { statusCache = saved; }
    }""", [text_model["id"], mixed])
    assert "the inspector runs Remek/basal-1.0-max" in other["blocked"]


def test_the_tool_builder_sets_the_selection_of_several_tools(page):
    view(page, "playground")
    tools = [{"name": "refund", "description": "zwrot"}, {"name": "track", "description": "śledzenie"}]
    request = {"state": "Chcę zwrotu i numeru przesyłki.", "questions": {"next": {
        "type": "tool", "tools": tools, "multi": True, "objective": "f1", "min": 1, "max": 2,
        "exclusive": ["refund"], "fallback": "track", "fallback_threshold": 0.3}}}
    page.evaluate("r => setRequest(r)", request)
    assert page.evaluate("() => buildRequest().questions") == request["questions"]
    card = page.locator(".question").first
    sync_api.expect(card.locator(".multi-tool-only")).to_be_visible()
    card.locator('[data-f="max"]').fill("1")
    card.locator('[data-f="objective"]').select_option("threshold")
    card.locator('[data-f="threshold"]').fill("0.6")
    built = page.evaluate("() => buildRequest().questions.next")
    assert built["max"] == 1 and built["threshold"] == 0.6 and "objective" not in built
    assert built["exclusive"] == ["refund"] and built["fallback"] == "track"
    card.locator('[data-f="multi"]').uncheck()
    sync_api.expect(card.locator(".multi-tool-only")).to_be_hidden()
    built = page.evaluate("() => buildRequest().questions.next")
    assert not {"multi", "min", "max", "exclusive", "fallback", "threshold"} & set(built)


def test_the_calibration_view_imports_a_file_fitted_elsewhere(page, ui, text_model, tmp_path):
    shipped = ui.get("/api/calibrations", params={"id": text_model["id"]})["shipped"]["file"]
    view(page, "models")  # the Calibration block (Loaded tab)
    page.select_option("#calibrationModel", text_model["id"])
    page.fill("#calibrationImportPath", shipped)
    sync_api.expect(page.locator("#calibrationFiles table")).to_be_visible()
    before = page.locator("#calibrationFiles .chip", has_text="imported").count()
    page.click("#importCalibration")
    sync_api.expect(page.locator("#calibrationImportPath")).to_have_value("")  # done: the field is cleared
    sync_api.expect(page.locator("#calibrationFiles .chip", has_text="imported")).to_have_count(before + 1)
    row = page.locator("#calibrationFiles tbody tr", has=page.locator(".chip", has_text="imported")).first
    sync_api.expect(row.locator("td").nth(3)).to_contain_text("5%")  # the shipped file's certified thresholds
    page.fill("#calibrationImportPath", str(tmp_path / "none.json"))
    page.click("#importCalibration")
    sync_api.expect(page.locator(".toast.error").last).to_contain_text("cannot read")


def test_theme_toggle_persists_and_sets_the_document_theme(page):
    page.click('[data-theme-choice="dark"]')
    assert page.evaluate("() => [document.documentElement.dataset.theme, localStorage.getItem('basal-ui.theme')]") == [
        "dark", "dark"]


def test_control_enter_sends_a_playground_decision(page):
    view(page, "playground")
    page.evaluate("""() => {
      api = async (path, options) => {
        window.sentDecision = path === '/api/decide' && options.method === 'POST';
        return {answers: {q: {type: 'choice', choice: 'yes'}}, usage: {}};
      };
      $('#editJsonToggle').checked = true;
      $('#requestJson').value = JSON.stringify({state: 'state', questions: {q: {type: 'choice', criteria: ['yes']}}});
    }""")
    page.locator("body").click(position={"x": 5, "y": 5})
    page.keyboard.press("Control+Enter")
    page.wait_for_function("() => window.sentDecision === true")


def test_question_mark_opens_shortcut_help(page):
    page.locator("body").click(position={"x": 5, "y": 5})
    page.keyboard.press("?")
    sync_api.expect(page.locator("#shortcutsDialog")).to_be_visible()
    sync_api.expect(page.locator("#shortcutsDialog")).to_contain_text("Ctrl/⌘ + Enter")


def test_alt_digit_opens_the_nth_page_by_key_code(page):
    page.locator("body").click(position={"x": 5, "y": 5})
    page.keyboard.press("Alt+2")
    sync_api.expect(page.locator("#pageTitle")).to_have_text("Models")
    # macOS Option+3 types "£" / "#": the key changes with the layout, the code does not
    page.evaluate("""() => document.body.dispatchEvent(new KeyboardEvent('keydown',
      {key: '£', code: 'Digit3', altKey: true, bubbles: true}))""")
    sync_api.expect(page.locator("#pageTitle")).to_have_text("Model manager")


def test_model_card_dialog_renders_api_card_data(page):
    page.evaluate("""() => {
      api = async () => ({
        id: 'community/model', repo: 'community/model', url: 'https://huggingface.co/community/model',
        label: 'Model label', author: 'community', community: true, base_repo: 'base/model',
        family: 'basal-2.0', tier: 'mini', port: 'MLX', size_bytes: 1000, media: [],
        downloaded: true, revisions: ['abc'], local_dir: '/models/model', basal_json: {prompt_format: 'chat'},
        shipped_calibration: {thresholds: {'0.01': {choice: 0.9}}},
        assessment: {mode: 'mlx', runnable: true, fit: 'fits', fit_now: 'fits', est_memory_bytes: 2000},
        loaded: null, readme: 'A model README.'
      });
      openModelCard('community/model');
    }""")
    sync_api.expect(page.locator("#modelCardDialog")).to_contain_text("Model label")
    sync_api.expect(page.locator("#modelCardDialog")).to_contain_text("base/model")
    sync_api.expect(page.locator("#modelCardDialog")).to_contain_text("A model README.")


def test_usage_line_includes_branches_and_lane(page):
    page.evaluate("""() => renderResults(
      {answers: {q: {type: 'choice', choice: 'yes'}},
       usage: {latency_ms: 1, branches: 3, second_order_branches: 2, lane: 'long'}},
      {model: selectedModel, questions: {q: {type: 'choice'}}}
    )""")
    line = page.locator("#results .answer-meta").first
    sync_api.expect(line).to_contain_text("branches: 3")
    sync_api.expect(line).to_contain_text("second_order_branches: 2")
    sync_api.expect(line).to_contain_text("lane: long")


def test_load_preset_save_and_apply_round_trip(page, text_model):
    view(page, "overview")
    page.click(f'#loadedModels [data-id="{text_model["id"]}"].options-model')
    dialog = page.locator("#loadOptions")
    sync_api.expect(dialog).to_be_visible()
    page.fill("#opt-revision", "test-revision")
    page.click("#saveLoadPreset")
    page.fill("#presetName", "e2e-load-preset")
    page.click("#presetNameSave")
    sync_api.expect(dialog).to_contain_text("e2e-load-preset")
    page.select_option("#loadPresetSelect", "e2e-load-preset")
    page.fill("#opt-revision", "")
    page.click("#applyLoadPreset")
    sync_api.expect(page.locator("#opt-revision")).to_have_value("test-revision")
    page.click("#deleteLoadPreset")
    sync_api.expect(dialog.locator("#loadPresetSelect")).not_to_contain_text("e2e-load-preset")
    page.click("#optionsCancel")


def test_revision_option_is_shown_as_a_loaded_model_chip(page, text_model):
    page.evaluate("""id => {
      const model = statusCache.models.find(x => x.id === id);
      model.options ||= {};
      model.options.revision = 'release-1';
      renderLoadedRows(statusCache.models);
    }""", text_model["id"])
    sync_api.expect(page.locator("#loadedModels")).to_contain_text("revision release-1")
    view(page, "models")
    page.evaluate("""id => renderModels({suggestions: [], models: modelData.map(m => m.id === id
      ? {...m, revision: 'release-1'} : m)})""", text_model["id"])
    row = page.locator("#modelsTable tr", has=page.locator(f'[data-id="{text_model["id"]}"]'))
    sync_api.expect(row.first).to_contain_text("revision release-1")
