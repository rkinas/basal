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
przykład `account` to „logowanie, hasło, kody weryfikacyjne, dane, adres, użytkownicy, usunięcie konta”. Na 300
zgłoszeniach podniosło to trafność kategorii z 68% do 87% (zob. „Wyniki”). Zrób to przed kalibracją, bo kalibracja
dotyczy dokładnie tych pytań, które serwujesz:

- pisz opisy w języku zgłoszeń;
- nazwij przypadki graniczne wprost, np. „zły rozmiar w dostawie to reklamacja”;
- nie zmieniaj kluczy odpowiedzi (`billing`, `account`, ...), bo używa ich Twoje kierowanie zgłoszeń.

## 5. Skalibruj na swoich zgłoszeniach

Model ma kalibrację dopasowaną na danych basala. W helpdesku model może być trafny, a mimo to inaczej pewny siebie.
Wtedy dostarczony próg przepuszcza za mało odpowiedzi: tutaj 1%. `basal-calibrate` dopasowuje temperatury i progi
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

**Oba języki w jednym pliku.** Jeśli serwujesz pytania po polsku i po angielsku, dodaj `--lang both`:

```bash
python examples/triage/run.py --rules --lang both --calibration-input labelled.jsonl   # 2 żądania na zgłoszenie
```

Każde zgłoszenie trafia do pliku dwa razy, raz z pytaniami po polsku i raz po angielsku:

- obie kopie mają wspólną `group`, więc lądują po tej samej stronie podziału FIT/CERT i liczby z CERT zostają
  uczciwe;
- każdy język dostaje własne `family`, więc raport podaje błąd kalibracji osobno dla każdego języka
  (`LOFO ECE`).

**Krok 3. Przeczytaj raport.**

| linia | na co patrzeć |
|---|---|
| `accuracy` per typ | jak często odpowiedź jest trafna, bez żadnego progu |
| `ECE on CERT shipped -> fitted` | błąd kalibracji. Im niżej, tym lepiej; dopasowanie powinno go obniżyć |
| `automated on CERT`, `max_error 0.05` | **główna liczba**: jaką część basal skieruje automatycznie i z jakim błędem, na zgłoszeniach, których dopasowanie nie widziało |
| `0.05 per type: choice 0.93 (35%)` | próg każdego typu pytania; `none (defers)` znaczy, że ten typ zawsze idzie do człowieka |
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

W basalu 2.5.2 `basal-run --preset` kończy każdą linię błędem `KeyError: 'q'`; poprawka to jedna linia w
`basal/run.py`. `basal-run --preset` wysyła preset bez `RULES`. Z własnymi regułami zapisz linie żądań (robi to `run.py
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

Pomiary na Apple M1 Max, basal 2.5.2, basal-2.0-4.5B-MLX-8bit, `--lang pl`.

- **Zgłoszenia:** 300. Zgłoszenia 1-100 są krótkie i jasne. Zgłoszenia 101-300 pochodzą z czterech rodzajów firm:
  sklepu internetowego, oprogramowania B2B, telekomu i mieszanych, chaotycznych e-maili (przekazane wątki, wielkie
  litery, brak polskich znaków, dwie sprawy w jednej wiadomości).
- **Przebieg:** około 7,5 minuty dla 300 zgłoszeń, czyli około 1,5 s na zgłoszenie, przy 8 żądaniach naraz.
- **Kalibracja:** około 12 minut z `basal-calibrate`.

### Trafność

| zgłoszenia | pytania | kategoria | pilność | człowiek? |
|---|---|---|---|---|
| 1-300 | preset | 68% | 65% | 77% |
| 1-300 | `--rules` | 87% | 85% | 79% |
| 1-100 | preset | 70% | 66% | 80% |
| 1-100 | `--rules` | 91% | 80% | 81% |

Model mini (basal-2.0-mini-MLX-8bit) na zgłoszeniach 1-40, zanim dodano `--lang pl`:

| pytania | kategoria | pilność | człowiek? |
|---|---|---|---|
| preset | 82% | 55% | 52% |
| `--rules` | 95% | 55% | 72% |

### Automatyzacja przy 5% błędu (`--rules --max-error 0.05`)

| kalibracja | kategoria | pilność | człowiek? |
|---|---|---|---|
| dostarczona (`CALIBRATION.json`) | 1% automatycznie | brak progu | brak progu |
| dopasowana na 300 zgłoszeniach, część CERT (88 zgłoszeń, których dopasowanie nie widziało) | **35% automatycznie, 0 błędów na 31** (próg 0.934) | **42% automatycznie, 1 błąd na 37** (próg 0.907) | brak progu |
| ten sam plik na serwerze, wszystkie 300 zgłoszeń | 38%, 0 błędów na 114 | 45%, 1% błędów na 135 | 0% |
| dopasowana tylko na zgłoszeniach 1-100, część CERT | 73%, 1 błąd na 22 (próg 0.929) | brak progu | brak progu |

Co z tego wynika:

- **Z własną kalibracją basal kieruje około jednej trzeciej zgłoszeń do właściwego zespołu, bez błędów na
  odłożonych zgłoszeniach.** Ustawia też pilność czterech na dziesięć. Kalibracja dostarczona z modelem nie
  automatyzuje w tym helpdesku nic.
- **Uczciwe są wiersze CERT.** Wiersz „wszystkie 300” obejmuje zgłoszenia, na których dopasowano progi.
- **Więcej danych daje mniejszy, ale prawdziwszy odsetek.** Na 100 łatwych zgłoszeniach CERT dawał 73%. Na 300, z
  chaotycznymi, daje 35%.
- **Progi są dopasowane, ale jeszcze niepotwierdzone przy 5%.** `basal-calibrate` ostrzega, że w CERT jest 31 i 37
  zaakceptowanych decyzji. Potwierdzenie 5% przy 0 błędach wymaga 59. To razem około 500 oznaczonych zgłoszeń.
- **`needs_human` nie dostaje progu przy żadnej liczbie danych w tym przykładzie.** To, kto musi decydować, jest
  kwestią polityki firmy, a model nie jest tu wystarczająco pewny. Zostaw to pytanie ludziom albo rozbij je na
  konkretne pytania tak/nie, np. „czy klient żąda zwrotu pieniędzy?”.
- **Dopasowanie obniża błąd kalibracji (ECE na CERT)**, najbardziej dla pilności:

  | pytanie | dostarczona | dopasowana |
  |---|---|---|
  | pilność | 0.184 | 0.027 |
  | kategoria | 0.089 | 0.046 |
  | człowiek? | 0.086 | 0.055 |

- **Większość pomyłek w kategorii to zgłoszenia wysłane do `other`.** W przebiegu na serwerze trafiło tam 12
  reklamacji i 11 zgłoszeń technicznych. Następny krok to ostrzejszy opis `other` w `RULES`. Po zmianie trzeba
  kalibrować ponownie.

### NVIDIA GB10, `--mode fast`, pytania polskie i angielskie w jednej kalibracji

basal-2.0-4.5B w bf16, `--rules --max-error 0.05`, na tych samych 300 polskich zgłoszeniach. Kalibrację
dopasowano z `--lang both`: 600 żądań, około 9 minut.

Raport kalibracji, część CERT (176 żądań, których dopasowanie nie widziało):

| pytanie | trafność | automatycznie | błędy wśród automatycznych | ECE dostarczona -> dopasowana |
|---|---|---|---|---|
| kategoria | 88% | 37.5% (próg 0.938) | 1 na 66 | 0.093 -> 0.042 |
| pilność | 83% | 52.8% (próg 0.863) | **5 na 93 (5.4%)** | 0.169 -> 0.019 |
| człowiek? | 80% | brak progu | - | 0.081 -> 0.062 |

Błąd kalibracji osobno dla każdego języka (`LOFO ECE`) wynosi 0.059-0.065 dla kategorii i około 0.045 dla pilności.
Jeden plik obsługuje oba języki.

Przebiegi na serwerze, wszystkie 300 zgłoszeń:

| kalibracja | `--lang` | kategoria: trafność, automatycznie, błędy | pilność | człowiek? |
|---|---|---|---|---|
| dostarczona | pl | 87%, 24%, 0 na 73 | 84%, brak progu | 79%, 9%, 0 na 27 |
| dostarczona | en | 89%, 17%, 0 na 51 | 81%, brak progu | 80%, 8%, 1 na 23 |
| dopasowana (both) | pl | 87%, 40%, 2% ze 121 | 84%, 54%, 4% ze 163 | 79%, brak progu |
| dopasowana (both) | en | 89%, 41%, 2% ze 122 | 81%, 57%, 3% ze 171 | 80%, brak progu |

Co się zmienia na CUDA:

- **Dostarczona kalibracja modelu bf16 już automatyzuje część zgłoszeń.** Kieruje 17-24% kategorii i 8-9%
  odpowiedzi `needs_human`, bez błędów albo z jednym. Na Macu dostarczony plik portu MLX automatyzuje około 1%.
  Każdy silnik i port ma własny plik kalibracji, więc mierz na tym, który serwujesz.
- **Dopasowany plik podwaja odsetek dla kategorii do około 40%,** w obu językach.
- **Dopasowany próg pilności nie trzyma 5%.** `basal-calibrate` ostrzega, że na CERT przyjmuje 93 odpowiedzi z 5
  błędami (5.4%), czyli powyżej celu 5%: „the threshold does not hold on CERT; label more data before relying on
  it”. Z tym plikiem nie kieruj zgłoszeń według pilności.
- **Dopasowany plik traci próg `needs_human`, który miał plik dostarczony.** Na tych zgłoszeniach żaden poziom
  pewności nie spełnił warunku. Dla tego pytania zostaw zachowanie dostarczonego pliku albo kieruj je do ludzi.
- **Pytania po angielsku działają na polskich zgłoszeniach tak samo dobrze jak po polsku:** 89% wobec 87% dla
  kategorii. Używaj języka, który czyta Twój zespół. Jedna kalibracja z `--lang both` obsłuży oba.

## Ograniczenia

- **Dla polskich zgłoszeń podawaj `--lang pl`.** Bez tej opcji język pytań dopasowuje się do zgłoszenia.
  `presets.lang_of` uznaje t33, „Aplikacja na iPhone wylogowuje mnie co kilka minut.”, za angielskie: nie ma polskich
  znaków i jest tylko jedno popularne polskie słowo.
- **Zgłoszenia napisano na potrzeby przykładu.** Prawdziwe są dłuższe i bardziej chaotyczne. Mierz na własnych.
- **`needs_human` zależy od Twojej polityki.** Opisy w `RULES` zapisują politykę jednego helpdesku. Napisz własne.
