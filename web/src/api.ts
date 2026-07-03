import type {
  BaselineComparisonResponse,
  ModelOption,
} from "./types";

const env = (import.meta as unknown as { env?: Record<string, string | undefined> }).env ?? {};
const apiBaseUrl = normalizeBaseUrl(env.VITE_API_URL ?? "");

function apiUrl(path: string) {
  return `${apiBaseUrl}${path}`;
}

function explainUrl() {
  return apiUrl("/predict");
}

function normalizeBaseUrl(value: string) {
  return value.replace(/\/+$/, "");
}

export async function fetchModels(): Promise<ModelOption[]> {
  const response = await fetch(apiUrl("/api/models"));
  if (!response.ok) {
    throw new Error(`Models request failed: ${response.status}`);
  }
  return response.json();
}

export async function fetchBaselineComparison(options: {
  sentence1: string;
  sentence2: string;
  staticMode: boolean;
  baselines: ModelOption[];
  threshold?: number;
}): Promise<BaselineComparisonResponse> {
  const response = await fetch(explainUrl(), {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({
      sentence1: options.sentence1,
      sentence2: options.sentence2,
      baselines: options.baselines,
      threshold: options.threshold,
      static: options.staticMode,
      allow_static_fallback: false
    })
  });
  if (!response.ok) {
    let detail = "";
    try {
      const body = await response.json();
      detail = body.detail ? `: ${body.detail}` : "";
    } catch {
      detail = "";
    }
    throw new Error(`Baseline comparison failed: ${response.status}${detail}`);
  }
  return response.json();
}
