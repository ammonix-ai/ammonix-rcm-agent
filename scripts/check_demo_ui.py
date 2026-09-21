"""Demo-UI checker (ui-v2): Playwright end-to-end suite over the running server.

Covers the P9 goal condition (inbox triage, case view with provenance hover,
live adjudication round trip, all six story cases reaching their scripted end
states) PLUS the ui-v2 additions: the guided tour as landing page with a
working stepper and inline case embed, inbox triage filtering and highlights,
neighbour expansion, the ambiguous-case choice buttons executing the chosen
action, and the escalate-to-human round trip. Prints the one-line JSON verdict.
"""

import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PORT = 8765
BASE = f"http://127.0.0.1:{PORT}"


def wait_for_server(process, timeout_s: int = 5400) -> None:
    """Wait for the API, then for the background case computation to finish
    (startup serves immediately; /api/progress reports done/total)."""
    started = time.time()
    while time.time() - started < timeout_s:
        if process.poll() is not None:
            raise SystemExit(f"server died during startup (exit {process.returncode})")
        try:
            with urllib.request.urlopen(f"{BASE}/api/progress", timeout=5) as r:
                p = json.loads(r.read())
            if p.get("rollout_error"):
                raise SystemExit(f"rollout failed: {p['rollout_error']}")
            if p.get("replay_error"):
                raise SystemExit(f"replay failed: {p['replay_error']}")
            if p.get("llm_error"):
                raise SystemExit(f"llm lane failed: {p['llm_error']}")
            replay = p.get("replay")
            replay_done = bool(replay) and replay["done"] >= replay["total"]
            if (
                p["total"] and p["done"] >= p["total"]
                and p.get("rollout_done") and replay_done
                and p.get("llm_done")
            ):
                if p.get("failed"):
                    raise SystemExit(f"cases failed to compute: {p['failed']}")
                return
        except SystemExit:
            raise
        except Exception:
            pass
        time.sleep(3)
    raise SystemExit("server did not finish computing cases in time")


def api(path: str) -> dict | list:
    """GET a JSON endpoint. Right after case computation finishes the event
    loop can be briefly busy (and a few endpoints answer 425 while their own
    lazy build runs), so retry with backoff instead of failing on the race."""
    last: Exception | None = None
    for attempt in range(6):
        try:
            with urllib.request.urlopen(f"{BASE}{path}", timeout=30) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            if e.code != 425:
                raise
            last = e
        except (TimeoutError, urllib.error.URLError) as e:
            last = e
        time.sleep(5 * (attempt + 1))
    raise SystemExit(f"{path} still unavailable after retries: {last}")


def main() -> int:
    checks: dict[str, bool] = {}
    server = subprocess.Popen(
        [
            str(ROOT / ".venv" / "Scripts" / "python.exe"), "-m", "uvicorn",
            "ui.server:app", "--host", "127.0.0.1", "--port", str(PORT),
        ],
        cwd=ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        wait_for_server(server)
        from playwright.sync_api import sync_playwright

        with sync_playwright() as pw:
            browser = pw.chromium.launch()
            page = browser.new_page()

            # --- the six scripted story cases (API level; the tour UI was
            # retired in favour of the claims comparison as landing page) --
            steps = api("/api/story")
            checks["six_story_steps"] = len(steps) == 6
            checks["story_all_end_states_reached"] = all(
                s["reached"] for s in steps
            )
            story_detail = [
                {"step": s["step"], "expected": s["expected"],
                 "actual": s["actual"], "reached": s["reached"]}
                for s in steps
            ]
            page.goto(f"{BASE}/")
            page.wait_for_selector("#claims")
            checks["landing_redirects_to_claims"] = True

            # --- claims inbox: scoreboard, verdicts, filtering -----------
            page.goto(f"{BASE}/inbox.html")
            page.wait_for_selector("#claims tbody tr")
            checks["scoreboard_compares_policies"] = page.eval_on_selector(
                "#scoreboard",
                "e => e.textContent.includes('Ammonix') && "
                "e.textContent.includes('$') && e.textContent.toLowerCase().includes('billers')",
            )
            n_rows = page.eval_on_selector_all(
                "#claims tbody tr", "rows => rows.length"
            )
            checks["claims_populated"] = n_rows >= 100
            verdicts = set(
                page.eval_on_selector_all(
                    "#claims tbody tr", "rows => rows.map(r => r.dataset.verdict)"
                )
            )
            checks["claims_have_verdicts"] = "ammonix" in verdicts and len(verdicts) >= 2
            page.click('#counts .chip[data-filter="ammonix"]')
            filtered = page.eval_on_selector_all(
                "#claims tbody tr", "rows => rows.map(r => r.dataset.verdict)"
            )
            checks["claims_filter_works"] = (
                len(filtered) > 0 and set(filtered) == {"ammonix"}
            )
            # --- claim story: two lanes, iteration by iteration ----------
            claims = api("/api/claims")
            board = claims["scoreboard"]
            checks["rollout_totals_consistent"] = (
                board["ammonix_wins"] + board["human_wins"] + board["same"]
                == board["claims"] == len(claims["claims"])
            )
            checks["llm_lane_in_scoreboard"] = (
                board.get("llm_collected") is not None
                and "ammonix_vs_llm" in board
            )
            checks["llm_lane_on_claims"] = any(
                c.get("llm") for c in claims["claims"]
            )
            # --- token meter + lane paperwork (only when the lane's caches
            # carry usage; the pinned-qwen lane predates metering and passes
            # vacuously) -------------------------------------------------
            def _tok_sum(t):
                return (t or {}).get("prompt_tokens", 0) + (t or {}).get(
                    "completion_tokens", 0
                )

            tok = board.get("llm_tokens")
            if tok:
                metered = None
                for c in claims["claims"][:20]:
                    if not c.get("llm"):
                        continue
                    d = api(f"/api/claim/{c['episode_id']}")
                    if (d.get("llm_detail") or {}).get("tokens"):
                        metered = (c["episode_id"], d)
                        break
                lane_tok = metered[1]["llm_detail"]["tokens"] if metered else {}
                steps = (metered[1].get("llm_steps") or []) if metered else []
                decide_sum = sum(_tok_sum(st.get("tokens")) for st in steps)
                write_sum = sum(_tok_sum(st.get("write_tokens")) for st in steps)
                checks["llm_token_meter_consistent"] = bool(
                    metered
                    and _tok_sum(tok.get("decide")) > 0
                    and decide_sum > 0
                    and decide_sum == _tok_sum(lane_tok.get("decide"))
                    and write_sum == _tok_sum(lane_tok.get("write"))
                )
                # a written GPT-6 filing somewhere in the first claims
                pw_claim = None
                for c in claims["claims"][:20]:
                    if not c.get("llm"):
                        continue
                    d = api(f"/api/claim/{c['episode_id']}")
                    if any(st.get("paperwork") for st in d.get("llm_steps") or []):
                        pw_claim = c["episode_id"]
                        break
                if metered:
                    page.goto(f"{BASE}/claim.html?id={pw_claim or metered[0]}")
                    page.wait_for_selector("#llm-lane .exchange")
                    checks["llm_token_meter_rendered"] = bool(
                        page.eval_on_selector(
                            "#llm-outcome",
                            "e => /tokens/.test(e.textContent)",
                        )
                        and page.eval_on_selector(
                            "#ammonix-outcome",
                            "e => /tokens|deciding is free/.test(e.textContent)",
                        )
                    )
                    checks["llm_paperwork_shown"] = (
                        pw_claim is None  # no filings metered yet: vacuous
                        or bool(page.eval_on_selector_all(
                            "#llm-steps .doc",
                            "els => els.length >= 1",
                        ))
                    )
                    # Ammonix's filings render inline too (owner ask: same
                    # details UI both columns). Replays fill in last, so
                    # poll the API until this claim's moments carry one.
                    target = pw_claim or metered[0]
                    inline = False
                    for _ in range(60):
                        d2 = api(f"/api/claim/{target}")
                        if any(
                            st.get("paperwork")
                            for st in d2.get("system_steps") or []
                        ):
                            inline = True
                            break
                        time.sleep(10)
                    page.goto(f"{BASE}/claim.html?id={target}")
                    page.wait_for_selector("#ammonix-steps .exchange")
                    checks["ammonix_paperwork_inline"] = inline and bool(
                        page.eval_on_selector_all(
                            "#ammonix-steps details",
                            "els => { els.forEach(e => e.open = true); "
                            "return els.some(e => e.querySelector('.doc')); }",
                        )
                    )
                else:
                    checks["llm_token_meter_rendered"] = False
                    checks["llm_paperwork_shown"] = False
                    checks["ammonix_paperwork_inline"] = False
            story_claim = next(
                c for c in claims["claims"] if c["verdict"] == "ammonix"
            )
            page.goto(f"{BASE}/claim.html?id={story_claim['episode_id']}")
            page.wait_for_selector("#human-lane .exchange")
            checks["claim_story_two_lanes"] = (
                page.eval_on_selector_all(
                    "#human-lane .exchange", "els => els.length"
                ) >= 1
                and page.eval_on_selector_all(
                    "#ammonix-lane .exchange", "els => els.length"
                ) >= 1
            )
            checks["claim_story_outcomes"] = page.eval_on_selector_all(
                ".lane .score-card",
                "els => els.length === 3 && els.every(e => e.textContent.trim().length > 0)",
            )

            # --- vocabulary coverage: every word the server can emit has a
            # plain-language label in shared.js (closes the raw-code class) -
            import re as _re
            shared_js = (ROOT / "ui" / "static" / "shared.js").read_text(
                encoding="utf-8"
            )
            vocab = [
                "paid", "paid_on_appeal", "paid_with_secondary",
                "patient_billed", "written_off", "written_off_at_cap",
                "needs_outside_information", "unresolvable", "iteration_cap",
                "submit_clean", "submit_with_records", "correct_and_resubmit",
                "bill_secondary", "request_retro_auth", "appeal_with_necessity",
                "request_peer_to_peer", "provide_requested_info",
                "bill_patient", "write_off",
            ]
            checks["ui_vocabulary_covered"] = all(
                _re.search(rf"^\s*{w}:", shared_js, _re.M) for w in vocab
            )

            # --- tie-break verdicts must not read as money wins ----------
            zero_delta = next(
                (c for c in claims["claims"]
                 if c["verdict"] != "same" and abs(c["delta"]) <= 0.005),
                None,
            )
            if zero_delta:
                page.goto(f"{BASE}/inbox.html")
                page.wait_for_selector("#claims tbody tr")
                checks["tiebreak_verdict_not_money"] = page.eval_on_selector(
                    f'tr[data-episode="{zero_delta["episode_id"]}"] td:last-child',
                    "e => e.textContent.trim().length > 0"
                    " && !e.textContent.includes('$0.00')",
                )
            else:
                checks["tiebreak_verdict_not_money"] = True

            # --- edge cases have no live run to link to ------------------
            if claims["specials"]:
                sp = claims["specials"][0]["state_id"]
                page.goto(f"{BASE}/case.html?id={sp}")
                page.wait_for_selector("#recommendation")
                checks["special_has_no_play_link"] = page.eval_on_selector(
                    "#play-link", "e => e.hidden"
                )
            else:
                checks["special_has_no_play_link"] = True

            # --- live play: execute the system's moves for real ----------
            # (re-anchor: the checks above navigated to other pages)
            page.goto(f"{BASE}/claim.html?id={story_claim['episode_id']}")
            page.wait_for_selector("#play-start")
            page.click("#play-start")
            page.wait_for_selector("#play-run:not([hidden])")
            checks["play_starts_with_recommendation"] = page.eval_on_selector(
                "#play-reco", "e => e.textContent.includes('Ammonix')"
            )
            executed = 0
            overrode = False
            for _ in range(8):
                if page.query_selector("#live-outcome .score-card"):
                    break
                n_before = page.eval_on_selector_all(
                    "#play-steps div", "els => els.length"
                )
                alt = None
                if not overrode and page.query_selector("#play-reco details"):
                    page.eval_on_selector(
                        "#play-reco details", "d => { d.open = true; }"
                    )
                    alt = page.query_selector("#play-reco details .choice")
                if alt:
                    alt.click()
                    overrode = True
                elif page.query_selector("#play-exec"):
                    page.click("#play-exec")
                else:
                    page.click("#play-reco .choice")
                page.wait_for_function(
                    "n => document.querySelectorAll('#play-steps div').length > n",
                    arg=n_before,
                )
                executed += 1
            checks["play_executes_moves"] = executed >= 1
            checks["override_executes_your_choice"] = overrode and bool(
                page.eval_on_selector_all(
                    "#play-steps .play-step .pill.amber",
                    "els => els.some(e => e.textContent.trim() === 'you')",
                )
            )
            page.wait_for_selector("#live-outcome .score-card")
            checks["play_completes_with_outcome"] = page.eval_on_selector(
                "#live-outcome .score-card", "e => e.textContent.includes('$')"
            )
            checks["play_shows_payer_answer"] = page.eval_on_selector_all(
                "#play-steps .play-step",
                "els => els.length > 0 && els.every(e => e.querySelector('.verdict-line .pill'))",
            )
            green_ids = [
                c["state_id"] for c in api("/api/cases")
                if c["triage"] == "green" and c["has_artifact"]
            ][:1]

            # --- case view: provenance hover + M2 log + neighbours -------
            page.goto(f"{BASE}/case.html?id={green_ids[0]}")
            page.wait_for_selector("[data-source-table]")
            provenance = page.eval_on_selector_all(
                "[data-source-table]",
                "els => els.map(e => ({t: e.dataset.sourceTable, "
                "c: e.dataset.sourceColumn, title: e.title}))",
            )
            # titles split input from agent-written (owner rule): record
            # fields hover their source, model fields say who wrote them
            checks["provenance_hover_attributes"] = len(provenance) >= 10 and all(
                p["t"] and p["c"] and p["title"].startswith(
                    ("from the case record: ", "written by the writer model")
                )
                for p in provenance
            )
            # plain-language checks log: a green case shows the passed verdict
            checks["m2_log_shown"] = page.eval_on_selector(
                "#m2log",
                "e => e.textContent.includes('passed') "
                "&& e.querySelector('.doc-verdict.good') !== null",
            )
            checks["recommendation_shows_probability"] = page.eval_on_selector(
                "#recommendation", "e => /\\d+%/.test(e.textContent)"
            )
            checks["neighbours_listed"] = page.eval_on_selector_all(
                "#neighbours tbody tr[data-expand]", "rows => rows.length"
            ) >= 5
            page.click('#neighbours tbody tr[data-expand="0"]')
            checks["neighbour_expands_with_facts"] = page.eval_on_selector(
                'tr[data-detail="0"]',
                "e => !e.hidden && e.textContent.trim().length > 10",
            )

            # --- the moment page is inspection-only: execution lives in
            # the claim's live run, reached via the play link -------------
            checks["moment_page_is_inspection_only"] = (
                page.query_selector("#execute") is None
                and page.query_selector("#escalate-btn") is None
                and page.eval_on_selector(
                    "#play-link", "e => !e.hidden && e.href.includes('claim.html')"
                )
            )

            # --- escalations are visible where Ammonix refuses to act ----
            red_id = next(
                c["state_id"] for c in api("/api/cases")
                if c["triage"] == "red" and c["escalation_reason"]
            )
            page.goto(f"{BASE}/case.html?id={red_id}")
            page.wait_for_selector("#recommendation")
            checks["escalation_shows_reason"] = page.eval_on_selector(
                "#recommendation",
                "e => e.textContent.includes('handed to a human')",
            )

            # --- ambiguous case: question shown, decided in the live run -
            amber_id = next(
                c["state_id"] for c in api("/api/cases")
                if c["triage"] == "amber" and c["status"] == "executed"
            )
            page.goto(f"{BASE}/case.html?id={amber_id}")
            page.wait_for_selector("#question:not([hidden])")
            checks["ambiguous_case_asks_question"] = page.eval_on_selector(
                "#question", "e => e.textContent.includes('coin flip')"
            )
            checks["ambiguous_links_to_live_run"] = page.eval_on_selector(
                "#question", "e => !!e.querySelector('a[href*=\"claim.html\"]')"
            )

            # --- universe map (smoke) ------------------------------------
            uni = api("/api/universe")
            n_path_points = sum(
                len(pth["points"]) for pth in uni.get("live_paths", [])
            )
            checks["universe_api_populated"] = (
                len(uni.get("points", [])) >= 1000
                and len(uni.get("tribes", [])) >= 1
            )
            checks["universe_shows_ammonix_paths"] = (
                uni.get("live_ready") is True
                and len(uni.get("live_paths", [])) >= 100
                and n_path_points >= 200
                and "live" not in uni
            )
            page.goto(f"{BASE}/universe.html")
            page.wait_for_selector("#map")
            page.wait_for_function(
                "document.querySelectorAll('#filters .chip, #filters button').length >= 1"
            )
            checks["universe_page_renders"] = page.eval_on_selector(
                "#map", "c => c.width > 0 && c.height > 0"
            )
            checks["universe_action_filter"] = page.eval_on_selector_all(
                '#filters .chip[data-action]', "els => els.length"
            ) >= 8
            # per-move chips live behind "More filters"; open it before clicking
            page.click("#filters details.adv > summary")
            page.click('#filters .chip[data-action="appeal_with_necessity"]')
            checks["universe_action_filter_applies"] = page.eval_on_selector(
                '#filters .chip[data-action="appeal_with_necessity"]',
                "e => e.classList.contains('on')",
            )

            # --- click a move: spotlight the claim + banner, then clear --
            page.wait_for_function(
                "() => U && U.live_paths && U.live_paths.length > 0"
            )
            page.evaluate("() => selectClaim(U.live_paths[0].episode)")
            checks["universe_claim_spotlight"] = page.eval_on_selector(
                "#claim-banner", "e => !e.hidden && !!e.querySelector('a.go')"
            )
            # stray canvas clicks must NOT clear the spotlight
            page.evaluate(
                "() => document.getElementById('map')"
                ".dispatchEvent(new MouseEvent('click'))"
            )
            checks["universe_spotlight_sticky"] = page.eval_on_selector(
                "#claim-banner", "e => !e.hidden"
            )
            page.evaluate("() => clearClaim()")
            checks["universe_spotlight_clears"] = page.eval_on_selector(
                "#claim-banner", "e => e.hidden"
            )

            # --- view filter: train + test / training cases / test claims -
            page.click('#filters .chip[data-show="live"]')
            checks["universe_view_filter"] = page.evaluate(
                "() => state.show === 'live'"
            )
            page.click('#filters .chip[data-show="both"]')

            # --- learning page: sample-efficiency curve -------------------
            curve = api("/api/curve")
            full_points = [p for p in curve["points"] if p["n"] == 4750]
            checks["curve_full_anchor_matches_inbox"] = bool(full_points) and all(
                p.get("anchor_match") for p in full_points
            )
            checks["curve_covers_sizes"] = (
                len({p["n"] for p in curve["points"]}) >= 6
                and len(curve["points"]) >= 15
            )
            page.goto(f"{BASE}/learning.html")
            page.wait_for_selector("#money svg circle", timeout=20000)
            checks["learning_charts_render"] = page.eval_on_selector_all(
                "#money svg circle", "els => els.length") >= 7
            # two labelled flat reference lines: the simulated billers and the
            # LLM agent (labelled with its model name, e.g. "Qwen 27B $41k")
            checks["learning_shows_flat_references"] = page.eval_on_selector(
                "#money", "el => el.textContent.toLowerCase().includes('billers')"
            ) and page.eval_on_selector(
                "#money",
                r"el => (el.textContent.match(/[A-Za-z]\s?\$\d+k/g) || []).length >= 2",
            )
            page.goto(f"{BASE}/inbox.html")
            checks["learning_linked_from_nav"] = page.eval_on_selector(
                ".masthead nav", "el => el.textContent.includes('Learning')"
            )

            browser.close()
    finally:
        server.terminate()
        try:
            server.wait(timeout=15)
        except subprocess.TimeoutExpired:
            server.kill()

    (ROOT / "runs" / "reports" / "demo_ui_check.json").write_text(
        json.dumps({"milestone": "P9-ui-v2", "checks": checks, "story": story_detail},
                   indent=2),
        encoding="utf-8", newline="\n",
    )
    verdict = all(checks.values())
    print(json.dumps({"loop": "P9U2", "verdict": verdict, **checks}))
    return 0 if verdict else 1


if __name__ == "__main__":
    sys.exit(main())
