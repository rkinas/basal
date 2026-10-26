// Request and response shapes of basal-serve's POST /v1/systemone (docs/API.md), the same as basal/client.py's.

export type Json = string | number | boolean | null | Json[] | { [key: string]: Json };

/** Options of a choice question: a map key -> description (null: the key is shown alone), or a list of keys. */
export type Criteria = Record<string, string | null> | string[];

export interface QuestionBase {
  instructions?: string;
  /** "show" (default): options are shown as "key: description"; "hide": the descriptions only. */
  option_keys?: "show" | "hide";
  /** Spans of the state that support the answer (models with an evidence head; not for multi; experimental). */
  evidence?: boolean;
  /** Answered only when every named question's answer is among the values; otherwise the answer is null. */
  ask_if?: Record<string, string[]>;
  /** Names of questions this one depends on (checked: known names, no cycle with ask_if). */
  depends_on?: string[];
}

/** Fields of choice, noul and score questions. */
export interface SetAndAbstain {
  /** Adds `answer_set` at this coverage (needs a calibration fitted with basal-calibrate --conformal). */
  coverage?: number;
  /** Adds `abstain` below the confidence threshold certified for this error and question type. */
  max_error?: number;
}

export interface ChoiceQuestion extends QuestionBase, SetAndAbstain {
  type: "choice";
  instructions: string;
  criteria: Criteria;
}

export interface NoulQuestion extends QuestionBase, SetAndAbstain {
  type: "noul";
  instructions: string;
  /** {true: "...", false: "..."}, optional. */
  criteria?: { true?: string; false?: string } | Record<string, string | null>;
}

export interface ScoreQuestion extends QuestionBase, SetAndAbstain {
  type: "score";
  instructions: string;
  /** Level texts, lowest first (or a map level key -> text). */
  criteria: string[] | Record<string, string>;
}

export interface MultiQuestion extends QuestionBase {
  type: "multi";
  instructions: string;
  criteria: Criteria;
  threshold?: number;
  min?: number;
  max?: number;
  objective?: "iou" | "f1" | "exact";
  exclusive?: string[];
  fallback?: string;
  fallback_threshold?: number;
}

export interface ActQuestion extends QuestionBase {
  type: "act";
  instructions: string;
  criteria: Criteria;
  /** {action: {outcome: cost}} or the short form {wrong: 99, defer: 1}. */
  costs: Record<string, number | Record<string, number>>;
  max_error?: number;
  defer_action?: string;
}

/** OpenAI (Chat Completions or Responses), MCP (inputSchema) or Anthropic (input_schema) tool definitions. */
export type ToolDefinition = { [key: string]: Json };

export interface ToolQuestion extends QuestionBase {
  type: "tool";
  tools: ToolDefinition[];
  /** Extra options after the tools; by default all three. */
  extra?: Array<"answer_directly" | "ask_user" | "none">;
  rank?: number;
  max_error?: number;
  description_chars?: number;
  /** The set of tools to call, using the multi objectives. */
  multi?: boolean;
  objective?: "iou" | "f1" | "exact";
  threshold?: number;
  min?: number;
  max?: number;
}

/** Task types (2.1, experimental; docs/TASKS.md): their fields are open, the answers are returned as documented there. */
export interface TaskQuestion extends QuestionBase {
  type: "classify" | "verify" | "compress" | "extract";
  [field: string]: unknown;
}

export type Question =
  | ChoiceQuestion
  | NoulQuestion
  | ScoreQuestion
  | MultiQuestion
  | ActQuestion
  | ToolQuestion
  | TaskQuestion;

export type Questions = Record<string, Question>;

export interface EvidenceSpan {
  text: string;
  start: number;
  end: number;
  probability: number;
}

/** Fields that any answer may carry. */
interface AnswerExtras {
  /** With "coverage": the keys of the answer set, the most probable first. */
  answer_set?: string[];
  coverage?: number;
  /** Why the set is every option (no threshold in the calibration for this level or type). */
  coverage_note?: string;
  /** With "max_error": true when the confidence is below the certified threshold. */
  abstain?: boolean;
  /** Present with "abstain" when no threshold is certified for the target (a note, not a number). */
  max_error?: string;
  evidence?: EvidenceSpan[];
  /** "unvalidated" when the calibration file does not belong to the served engine or option orders. */
  calibration?: "validated" | "unvalidated";
}

export interface ChoiceAnswer extends AnswerExtras {
  type: "choice";
  choice: string;
  probabilities: Record<string, number>;
  confidence: number;
}

export interface NoulAnswer extends AnswerExtras {
  type: "noul";
  /** Probability of yes. */
  noul: number;
  probabilities: { true: number; false: number };
  confidence: number;
}

export interface ScoreAnswer extends AnswerExtras {
  type: "score";
  /** The expected level (0 = the first). */
  score: number;
  legend: Record<string, string>;
  probabilities: Record<string, number>;
  confidence: number;
}

export interface MultiAnswer extends AnswerExtras {
  type: "multi";
  selected: string[];
  probabilities: Record<string, number>;
  threshold: number;
  set_confidence: number;
  expected_score?: number;
  objective?: string;
}

export interface ActAnswer extends AnswerExtras {
  type: "act";
  action: string;
  expected_costs: Record<string, number>;
  answer: string;
  probabilities: Record<string, number>;
  confidence: number;
  /** With max_error: true when the action is the defer action. */
  deferred?: boolean;
  /** With max_error: the automatic action that was refused. */
  refused?: string;
}

export interface ToolAnswer extends AnswerExtras {
  type: "tool";
  /** The tool name or extra option; null when deferred. */
  tool: string | null;
  call: boolean;
  probabilities: Record<string, number>;
  confidence: number;
  ranking?: Array<{ tool: string; probability: number }>;
  deferred?: boolean;
  refused?: string;
  /** With "multi": true: the tools to call (instead of "tool"). */
  tools?: string[];
}

/** Answers of the task types (docs/TASKS.md), which are not typed further here. */
export interface TaskAnswer extends AnswerExtras {
  type: string;
  [field: string]: unknown;
}

export type Answer = ChoiceAnswer | NoulAnswer | ScoreAnswer | MultiAnswer | ActAnswer | ToolAnswer | TaskAnswer;

/** The answer type of a question type. */
export interface AnswerOf {
  choice: ChoiceAnswer;
  noul: NoulAnswer;
  score: ScoreAnswer;
  multi: MultiAnswer;
  act: ActAnswer;
  tool: ToolAnswer;
}

export type AnswerFor<Q> = Q extends { type: infer T }
  ? T extends keyof AnswerOf
    ? AnswerOf[T]
    : TaskAnswer
  : Answer;

/** A question skipped by ask_if has the answer null. */
export type AnswersFor<Q extends Questions> = { [K in keyof Q]: AnswerFor<Q[K]> | null };

/** An image: a "data:image/..." URL, or {content_type, base64} (the form of vLLM's structured-diffusion example). */
export type ImageInput = string | { content_type: string; base64: string };

export interface MediaItem {
  type: "image" | "audio";
  /** Base64 (or a data URL). */
  data: string;
}

/** Request fields next to state and questions. */
export interface RequestOptions {
  facts?: "off" | "auto";
  think?: "auto" | "off" | "force";
  think_text?: boolean;
  early_exit?: "off" | "0.999" | "0.995" | "0.99" | "0.98";
  /** Image tokens 280 / 1,120 (with media or images). */
  detail?: "default" | "high";
  lang?: "pl" | "en";
  /** Images and audio (--media on servers). A request gives media or images, not both. */
  media?: MediaItem[];
  images?: ImageInput[];
  /** Any other field, sent as it is. */
  [field: string]: unknown;
}

export interface Usage {
  input_tokens: number;
  output_tokens: number;
  questions: number;
  branches: number;
  latency_ms: number;
  /** Names of the questions skipped by ask_if. */
  skipped?: string[];
  truncated?: boolean;
  second_order_branches?: number;
  [field: string]: unknown;
}

export interface DecisionResponse<A = Record<string, Answer | null>> {
  model: string;
  answers: A;
  usage?: Usage;
}

export interface ModelInfo {
  name: string;
  description?: string;
  release_date?: string;
  mode?: string;
  early_exit?: string[];
  temperature_per_prim?: Record<string, number>;
  prompt_format?: number;
  orders?: number;
  [field: string]: unknown;
}

export interface Health {
  status: string;
  running: number;
  queued: number;
}
