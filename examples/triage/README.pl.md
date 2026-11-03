# Triage zgłoszeń helpdesku z basalem

[English version](README.md)

## Przypadek użycia

Helpdesk sklepu internetowego dostaje setki wiadomości dziennie: reklamacje, pytania o faktury, awarie, reset hasła,
propozycje współpracy. Dziś każdą czyta człowiek i rozstrzyga trzy rzeczy:

- do którego zespołu ją przekazać: reklamacje, płatności, technika, konto czy inne;
- jak pilna jest sprawa;
- czy sprawę musi rozstrzygnąć człowiek, czy wystarczy gotowa odpowiedź.

basal odpowiada na te trzy pytania w jednym żądaniu i do każdej odpowiedzi podaje skalibrowaną pewność. Po
ustawieniu celu błędu (`max_error`) pewne odpowiedzi kierowane są automatycznie. Odpowiedź poniżej certyfikowanego
progu się wstrzymuje (abstain) i takie zgłoszenie trafia do człowieka. Sam wybierasz, jaki błąd akceptujesz, np. 5%.
Kalibracja na Twoich oznaczonych zgłoszeniach pokazuje, jaką część ruchu basal obsłuży przy tym błędzie.

```
helpdesk / e-mail / formularz ──> webhook (app.py) ──> basal-serve ──> odpowiedzi + pewność
                                         │
                                         ├─ kategoria certyfikowana       -> kolejka zespołu (automatycznie)
                                         ├─ kategoria wstrzymana          -> kolejka "human" (kieruje człowiek)
                                         ├─ najwyższy poziom pilności     -> priorytet "high"
                                         └─ needs_human tak / wstrzymane  -> sprawę rozstrzyga człowiek
```

## Pliki

| plik | co robi |
|---|---|
| `tickets.jsonl` | polskie zgłoszenia z poprawnymi etykietami: `{"id", "state", "gold": {"category", "urgency", "needs_human"}}` |
| `run.py` | puszcza preset `triage` przez zgłoszenia i liczy trafność. Opcje: `--rules` (reguły tego helpdesku), `--max-error` (automatyzacja), `--calibration-input` (dane wejściowe dla `basal-calibrate`) |
| `app.py` | webhook: `POST /ticket {"id", "text"}` zwraca kolejkę, priorytet i odpowiedzi |
| `../../tests/test_example_triage.py` | testy `run.py` i `app.py` bez modelu |

## 1. Uruchom serwer basala

Zainstaluj basala ([docs/INSTALL.md](../../docs/INSTALL.md)), potem uruchom serwer poleceniem dla swojego sprzętu:

```bash
# Mac (Apple Silicon), extra basal[mlx]
basal-serve --model Remek/basal-2.0-4.5B-MLX-8bit --mode mlx --port 8000

# GPU NVIDIA
basal-serve --model Remek/basal-2.0-4.5B --mode fast --port 8000

# słabsza maszyna: model mini (około 1.8 GB na Macu)
basal-serve --model Remek/basal-2.0-mini-MLX-8bit --mode mlx --port 8000
```

Sprawdź, czy działa:

```bash
curl -s localhost:8000/health       # {"status":"ok",...}
curl -s localhost:8000/v1/models    # model, jego temperatury i progi "certified"
```

## 2. Uruchom przykład

```bash
python examples/triage/run.py --lang pl                         # sam preset
python examples/triage/run.py --lang pl --rules                 # preset z regułami tego helpdesku
python examples/triage/run.py --lang pl --rules --max-error 0.05
python examples/triage/run.py --lang pl --rules --output answers.jsonl   # zapisz też wszystkie odpowiedzi
```

| opcja | znaczenie |
|---|---|
| `--url` | serwer basala (domyślnie `http://127.0.0.1:8000`) |
| `--lang pl\|en` | język pytań. Dla polskich zgłoszeń podawaj `pl` (zob. „Ograniczenia”) |
| `--rules` | pytania presetu z opisami odpowiedzi z `RULES` w `run.py` |
| `--max-error 0.05` | cel błędu: odpowiedzi poniżej certyfikowanego progu się wstrzymują |
| `--concurrency 8` | ile żądań naraz; serwer łączy je w paczki |
| `--input` | własne zgłoszenia w tym samym formacie |
| `--calibration-input PLIK` | zapisz oznaczone żądania dla `basal-calibrate` i zakończ; nie wymaga serwera |

Co znaczą kolumny wyniku:

- **accuracy**: odsetek odpowiedzi zgodnych z poprawną etykietą;
- **automated**: odsetek odpowiedzi, które się nie wstrzymały przy `--max-error`;
- **error when automated**: odsetek błędnych odpowiedzi wśród automatycznych. Kalibracja trzyma go na poziomie
  celu albo niżej;
- **automated end to end**: zgłoszenia, w których żadne pytanie się nie wstrzymało;
- **abstained ... no certified threshold**: kalibracja nie ma progu dla tego typu pytania przy tym błędzie, więc
  każda taka odpowiedź trafia do człowieka.

## 3. Przetestuj

```bash
# bez modelu: pytania, dane do kalibracji i webhook, z zaślepką basala
uv run --extra test python -m pytest -q tests/test_example_triage.py

# na żywo: cały przepływ na działającym serwerze
python examples/triage/run.py --lang pl --rules --max-error 0.05

# webhook na działającym serwerze
BASAL_URL=http://127.0.0.1:8000 uvicorn app:app --app-dir examples/triage --port 9000
curl -s localhost:9000/ticket -H 'content-type: application/json' \
     -d '{"id": "123", "text": "Pobrano ode mnie dwa razy 129 zł za abonament. Proszę o zwrot."}'
```

Webhook odpowiada tak (pole `answers` skrócone):

```json
{"id": "123", "queue": "billing", "priority": "normal", "category": "billing", "urgency": "średnia",
 "needs_human": true, "automated": false, "answers": {...}}
```

## 4. Wpisz swoje reguły w pytania

Preset to punkt wyjścia. `RULES` w `run.py` opisuje przy każdej odpowiedzi, co obejmuje w tym helpdesku. Na
przykład `account` to „logowanie, hasło, kody weryfikacyjne, dane, adres, użytkownicy, usunięcie konta”. Na 100
zgłoszeniach podniosło to trafność kategorii z 70% do 91% (zob. „Wyniki”). Zrób to przed kalibracją, bo kalibracja
dotyczy dokładnie tych pytań, które serwujesz:

- pisz opisy w języku zgłoszeń;
- nazwij przypadki graniczne wprost, np. „zły rozmiar w dostawie to reklamacja”;
- nie zmieniaj kluczy odpowiedzi (`billing`, `account`, ...), bo używa ich Twoje kierowanie zgłoszeń.

## 5. Skalibruj na swoich zgłoszeniach

Model ma kalibrację dopasowaną na danych basala. W helpdesku model może być trafny, a mimo to inaczej pewny siebie.
Wtedy dostarczony próg przepuszcza za mało odpowiedzi: tutaj 0%. `basal-calibrate` dopasowuje temperatury i progi
na Twoich oznaczonych zgłoszeniach.

**Krok 1. Oznacz zgłoszenia.** Weź prawdziwe zgłoszenia i wpisz poprawne odpowiedzi w `gold`, w formacie
`tickets.jsonl`. Ile ich potrzeba:

- **59** zaakceptowanych decyzji bez błędu na typ pytania w części FIT (70% danych) dla celu 5%;
- **299** dla celu 1%;
- mniej więcej tyle samo w części CERT (30%), która potwierdza próg.

W praktyce to 300+ zgłoszeń dla 5% i 1500+ dla 1%. Oznaczaj rzetelnie i bierz zgłoszenia z całego roku, nie z
jednego tygodnia.

**Krok 2. Zapisz dane wejściowe i dopasuj.**

```bash
python examples/triage/run.py --lang pl --rules --calibration-input labelled.jsonl
basal-calibrate labelled.jsonl --model Remek/basal-2.0-4.5B-MLX-8bit --mode mlx --out CALIBRATION.triage.json
# albo na działającym serwerze uruchomionym z --no-calibration:
basal-calibrate labelled.jsonl --url http://127.0.0.1:8000 --model Remek/basal-2.0-4.5B-MLX-8bit
```

Użyj tego samego modelu, trybu (`mlx`, `fast`, `gguf`, ...), kwantyzacji i pytań co na produkcji: progi należą do
nich.

**Krok 3. Przeczytaj raport.**

| linia | na co patrzeć |
|---|---|
| `accuracy` per typ | jak często odpowiedź jest trafna, bez żadnego progu |
| `ECE on CERT shipped -> fitted` | błąd kalibracji. Im niżej, tym lepiej; dopasowanie powinno go obniżyć |
| `automated on CERT`, `max_error 0.05` | **główna liczba**: jaką część basal skieruje automatycznie i z jakim błędem, na zgłoszeniach, których dopasowanie nie widziało |
| `0.05 per type: choice 0.93 (73%)` | próg każdego typu pytania; `none (defers)` znaczy, że ten typ zawsze idzie do człowieka |
| `warning: ... too few to confirm on CERT` | próg jest dopasowany, ale jeszcze niepotwierdzony. Oznacz więcej zgłoszeń |

`--report-only` wypisuje raport bez zapisywania pliku. `--save-rows rows.jsonl` zapisuje prawdopodobieństwa, a
`--from-rows` dopasowuje z nich ponownie bez modelu, np. dla innych `--targets`.

**Krok 4. Serwuj z kalibracją.**

```bash
basal-serve --model Remek/basal-2.0-4.5B-MLX-8bit --mode mlx --calibration CALIBRATION.triage.json --port 8000
curl -s localhost:8000/v1/models | python -m json.tool    # "certified" pokazuje teraz Twoje progi
python examples/triage/run.py --lang pl --rules --max-error 0.05
```

**Kalibruj ponownie**, gdy zmienia się model, tryb, kwantyzacja albo treść pytań, gdy pojawia się nowy rodzaj
zgłoszeń, albo co kilka miesięcy na świeżo oznaczonych zgłoszeniach.

## 6. Połącz z innymi aplikacjami

**Python (Twój backend)**, przez klienta `basal.client`:

```python
from basal.client import Basal
import run   # examples/triage/run.py: triage_questions, predicted

b = Basal("http://127.0.0.1:8000", retries=2)          # api_key="..." gdy serwer ma --api-keys
text = "Ktoś zalogował się na moje konto z obcego kraju."
answers = b.decide(text, run.triage_questions(text, "pl", max_error=0.05, rules=True))
cat = answers["category"]
queue = "human" if cat.get("abstain") else cat["choice"]
```

`AsyncBasal` ma te same metody dla asyncio (FastAPI, Starlette, aiohttp).

**Dowolny język przez HTTP**: wyślij pytania POST-em na `/v1/systemone`. Treść żądania jest taka jak w
`labelled.jsonl`, tylko bez `gold`, plus `max_error` przy każdym pytaniu ([docs/API.md](../../docs/API.md)):

```bash
curl -s localhost:8000/v1/systemone -H 'content-type: application/json' -d '{
  "state": "Pobrano dwa razy 129 zł.",
  "questions": {"category": {"type": "choice", "instructions": "Czego dotyczy zgłoszenie?",
                             "criteria": {"billing": "płatności, faktury, zwroty", "other": "coś innego"},
                             "max_error": 0.05}}}'
```

**Systemy helpdesk (Zendesk, Freshdesk, Jira Service Management, bramka pocztowa)**: skieruj ich webhook albo
automatyzację na `app.py` (`POST /ticket` z `{"id", "text"}`). Wynik zapisz z powrotem przez ich API: `queue` jako
grupę, `priority` jako priorytet, a `automated: false` jako tag „do sprawdzenia”. Uruchamiaj `app.py` obok
basal-serve, w sieci wewnętrznej.

**n8n, Make, Zapier**: węzeł HTTP Request, `POST http://<host>:9000/ticket` z JSON `{"id": "{{id}}", "text":
"{{body}}"}`, potem węzeł Switch po polu `queue`. Albo wołaj `/v1/systemone` bezpośrednio z treścią jak wyżej.

**Wsadowo: nocny przebieg po eksporcie**:

```bash
basal-run --preset triage --lang pl --input export.jsonl --output answers.jsonl --url http://127.0.0.1:8000/v1/systemone
```

`basal-run --preset` wysyła preset bez `RULES`. Z własnymi regułami zapisz linie żądań (robi to `run.py
--calibration-input`; usuń `gold`) albo użyj `run.py --input export.jsonl --output answers.jsonl`.

**Agenci (LangChain, LlamaIndex, CrewAI, MCP)**:

```python
from basal.adapters import langchain_tool
triage = langchain_tool("http://127.0.0.1:8000", preset="triage", lang="pl")   # agent podaje tylko treść zgłoszenia
```

Dla MCP `basal-serve --mcp on` udostępnia `/mcp`, a `basal-mcp` łączy z hostami, które uruchamiają lokalne procesy
([docs/MCP.md](../../docs/MCP.md)).

**Bezpieczeństwo.** Serwer jest domyślnie otwarty:

- `--api-keys keys.json` wymaga klucza na każdej trasie poza `/health` ([docs/SERVING.md](../../docs/SERVING.md#api-access));
- `--host 127.0.0.1`, gdy basala woła tylko `app.py` na tej samej maszynie;
- zgłoszenia zawierają dane osobowe, więc trzymaj basala i webhook w swojej sieci.

## 7. Na produkcji: zamknij pętlę

1. **Zacznij w trybie cienia.** basal odpowiada, ludzie nadal kierują każde zgłoszenie, a Ty porównujesz wyniki.
2. **Potem włącz `max_error`.** Certyfikowane odpowiedzi kieruj automatycznie, resztę do kolejki `human`.
3. **Oznaczaj dalej.** Ostateczną decyzję człowieka zapisuj jako nowe `gold` i sprawdzaj też próbkę zgłoszeń
   skierowanych automatycznie. Bez tej próbki widzisz tylko błędy w przypadkach, w których basal był niepewny.
4. **Kalibruj ponownie na nowych etykietach** co kilka miesięcy albo gdy odsetek automatyzacji lub błąd zaczyna się
   zmieniać.

`basal-serve --log-decisions decisions.jsonl` zapisuje log decyzji. `GET /metrics` zwraca metryki Prometheus.

## Wyniki

Pomiary na Apple M1 Max, basal 2.5.2. Przebieg przez 100 zgłoszeń na modelu 4.5B trwa około 40 s.

### Trafność

Wiersze 4.5B uruchomiono z `--lang pl`. Wiersze mini uruchomiono wcześniej, zanim dodano `--lang pl`, więc język
pytań dopasowywał się do każdego zgłoszenia.

| model | zgłoszenia | pytania | kategoria | pilność | człowiek? |
|---|---|---|---|---|---|
| basal-2.0-mini-MLX-8bit | 1-40 | preset | 82% | 55% | 52% |
| basal-2.0-mini-MLX-8bit | 1-40 | `--rules` | 95% | 55% | 72% |
| basal-2.0-4.5B-MLX-8bit | 1-100 | preset | 70% | 66% | 80% |
| basal-2.0-4.5B-MLX-8bit | 1-100 | `--rules` | 91% | 80% | 81% |

### Automatyzacja przy 5% błędu, basal-2.0-4.5B-MLX-8bit, 100 zgłoszeń

| kalibracja | kategoria automatycznie | błąd wśród automatycznych | pilność, człowiek? |
|---|---|---|---|
| dostarczona (`CALIBRATION.json`) | 0% (próg 0.977) | - | brak progu: wszystko się wstrzymuje |
| dopasowana na 100 zgłoszeniach | 83% (próg 0.929) | 1 z 83 | brak progu: wszystko się wstrzymuje |
| to samo, tylko część CERT | 73% | 1 z 22 | - |

- **Wynik 83% jest optymistyczny:** zmierzony na tych samych zgłoszeniach, na których dopasowano progi.
- **Uczciwa liczba to CERT:** 73% automatycznie, 1 błąd na 22.
- **Próg 5% nie jest jeszcze potwierdzony.** `basal-calibrate` zgłasza, że w CERT jest za mało decyzji.
- **Pilność i „człowiek?” nie dostają progu.** Za mało ich odpowiedzi jest jednocześnie pewnych i trafnych, więc
  te pytania zostają u ludzi.

Dopasowanie obniża błąd kalibracji (ECE na CERT):

| pytanie | dostarczona | dopasowana |
|---|---|---|
| kategoria | 0.122 | 0.087 |
| pilność | 0.140 | 0.071 |
| człowiek? | 0.126 | 0.121 |

## Ograniczenia

- **Dla polskich zgłoszeń podawaj `--lang pl`.** Bez tej opcji język pytań dopasowuje się do zgłoszenia.
  `presets.lang_of` uznaje t33, „Aplikacja na iPhone wylogowuje mnie co kilka minut.”, za angielskie: nie ma polskich
  znaków i jest tylko jedno popularne polskie słowo.
- **Zgłoszenia napisano na potrzeby przykładu.** Prawdziwe są dłuższe i bardziej chaotyczne. Mierz na własnych.
- **`needs_human` zależy od Twojej polityki.** Opisy w `RULES` zapisują politykę jednego helpdesku. Napisz własne.
