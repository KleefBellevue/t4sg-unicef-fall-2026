"""Usage: python pipeline/runner.py config/runs/<run>.yaml"""

import csv
import sys
import time
from pathlib import Path

import litellm
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent

RESULT_COLUMNS = [
    "id", "model", "mode", "thinking", "web_search", "repeat", "answer",
    "headline_number", "year_given", "source_claimed", "searched", "caveat",
    "number_score", "year_score", "source_score", "error_types",
    "input_tokens", "output_tokens", "cost", "seconds",
]


def mode_cold(prompt, run):
    return [{"role": "user", "content": prompt}]


def mode_skill(prompt, run):
    skill = (REPO_ROOT / run["skill_file"]).read_text(encoding="utf-8")
    return [{"role": "system", "content": skill}, {"role": "user", "content": prompt}]


MODES = {"cold": mode_cold, "skill": mode_skill}


TIMEOUT_SECONDS = 120
RETRYABLE = (litellm.Timeout, litellm.RateLimitError, litellm.ServiceUnavailableError,
             litellm.InternalServerError)


def ask(model, messages, thinking, web_search, log):
    kwargs = {"model": model["litellm_model"], "messages": messages}
    if model["supports_thinking"]:
        if model["provider"] == "gemini":
            kwargs["reasoning_effort"] = "low" if thinking else "disable"
        elif model["provider"] == "groq":
            kwargs["reasoning_effort"] = "default" if thinking else "none"
    if web_search:
        kwargs["tools"] = [{"googleSearch": {}}]

    for wait in [30, 60, 120, None]:
        start = time.time()
        try:
            response = litellm.completion(**kwargs, timeout=TIMEOUT_SECONDS)
            break
        except RETRYABLE as err:
            error = f"{type(err).__name__}: {str(err).splitlines()[0][:200]}"
            if wait is None:
                log(f"{error}, giving up")
                raise
            log(f"{error}, retrying in {wait}s")
            time.sleep(wait)
    seconds = round(time.time() - start, 2)

    input_tokens = response.usage.prompt_tokens or 0
    output_tokens = response.usage.completion_tokens or 0
    cost = (input_tokens * model.get("price_input_per_m", 0)
            + output_tokens * model.get("price_output_per_m", 0)) / 1e6
    grounding = getattr(response, "vertex_ai_grounding_metadata", None) or []
    searched = any(g.get("webSearchQueries") for g in grounding if isinstance(g, dict))

    return {
        "answer": response.choices[0].message.content or "",
        "searched": searched,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost": cost,
        "seconds": seconds,
    }


def main():
    with open(sys.argv[1]) as f:
        run = yaml.safe_load(f)

    prompts = []
    for path in run["prompts"]:
        with open(REPO_ROOT / path, newline="", encoding="utf-8") as f:
            prompts.extend(csv.DictReader(f))

    grid = [
        (p, name, model, mode, thinking, web, repeat)
        for repeat in range(1, run["repeats"] + 1)
        for p in prompts
        for name, model in run["models"].items()
        for mode in run["modes"]
        for thinking in run["thinking"]
        for web in run["web_search"]
        if (model["supports_thinking"] or not thinking)
        and (model["supports_web_search"] or not web)
    ]

    out_dir = REPO_ROOT / "results" / run["run_name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "results.csv", "w", newline="", encoding="utf-8") as f, \
         open(out_dir / "log.txt", "w", encoding="utf-8") as log_file:
        writer = csv.DictWriter(f, fieldnames=RESULT_COLUMNS, lineterminator="\n")
        writer.writeheader()
        spent = 0.0
        for n, (p, name, model, mode, thinking, web, repeat) in enumerate(grid, 1):
            if spent >= run["budget_usd"]:
                break
            label = (f"[{n}/{len(grid)}] {p['id']} {name} {mode} "
                     f"thinking={thinking} web_search={web} repeat={repeat}")
            print(label)

            def log(message):
                line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {label} {message}"
                print(line)
                log_file.write(line + "\n")
                log_file.flush()

            result = ask(model, MODES[mode](p["prompt"], run), thinking, web, log)
            spent += result["cost"]
            row = {c: "" for c in RESULT_COLUMNS}
            row.update({"id": p["id"], "model": name, "mode": mode, "thinking": thinking,
                        "web_search": web, "repeat": repeat, **result})
            writer.writerow(row)
            f.flush()


if __name__ == "__main__":
    main()
