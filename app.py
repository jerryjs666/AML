import json
import logging
from typing import Any, Dict, List

import streamlit as st

from agents import run_pipeline, parse_problem_payload
from rag_engine import ProblemDatabase

logging.basicConfig(level=logging.INFO)

st.set_page_config(page_title="APPS RAG Multi-Agent Demo", layout="wide")


def main():
    st.title("RAG + Multi-Agent APPS Solver")

    db = ProblemDatabase()
    db_ready = db.is_ready()
    if db_ready:
        db.load()
    else:
        st.warning(
            "Reference DB missing. Build it with:\n"
            "python scripts/build_reference_db.py --split test --limit 500\n"
            "python scripts/build_tfidf_index.py"
        )

    tabs = st.tabs(["Problem & Run", "War Room / Agent Debate", "RAG & Analytics"])

    with tabs[0]:
        st.subheader("Paste APPS Problem JSON")
        sample_text = st.session_state.get("last_payload", "")
        payload_text = st.text_area("Problem JSON", sample_text, height=220)

        provider = st.selectbox("Provider", ["DeepSeek", "OpenAI"], index=0)
        api_key = st.text_input("API Key", type="password")
        max_iters = st.number_input("Max iterations", min_value=1, max_value=6, value=3)
        rag_k = st.number_input("RAG top-k", min_value=1, max_value=10, value=3)
        timeout = st.number_input("Timeout per test (sec)", min_value=0.5, max_value=10.0, value=2.0, step=0.5)

        run_btn = st.button("Run Full Pipeline", type="primary")

        if run_btn:
            if not db_ready:
                st.error("RAG DB not built. Build artifacts before running.")
            else:
                try:
                    payload = parse_problem_payload(payload_text)
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Payload parsing error: {exc}")
                    st.stop()
                st.session_state["last_payload"] = payload_text
                retrieved = db.search(payload.get("question", ""), k=int(rag_k))
                retrieved_dicts = [r.__dict__ for r in retrieved]
                try:
                    results = run_pipeline(
                        payload_text=json.dumps(payload),
                        provider=provider,
                        api_key=api_key,
                        retrieved=retrieved_dicts,
                        max_iters=int(max_iters),
                        timeout=float(timeout),
                    )
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Pipeline failed: {exc}")
                    st.stop()
                st.session_state["run_results"] = results
                st.session_state["retrieved"] = retrieved_dicts

        if "run_results" in st.session_state:
            res = st.session_state["run_results"]
            st.success(f"Final Decision: {res['decision']}")
            st.code(res.get("final_code", ""), language="python")
            guide = res.get("guide")
            if guide:
                st.write("### Teacher-Style Guide")
                st.markdown(render_guide_markdown(guide))

    with tabs[1]:
        st.subheader("War Room")
        traces: List = []
        if "run_results" in st.session_state:
            traces = st.session_state["run_results"].get("traces", [])
        if not traces:
            st.info("Run the pipeline to see agent debate.")
        else:
            for t in traces:
                with st.expander(f"Iteration {t.iteration}"):
                    st.write("#### Retrieval")
                    st.json(t.retrieval)
                    st.write("#### Solver code")
                    st.code(t.solver.code, language="python")
                    st.write("Approach:", t.solver.approach)
                    st.write("#### Critic feedback (summary)")
                    st.markdown(render_critic_summary(t.critic))
                    with st.expander("Raw Critic JSON"):
                        st.json(t.critic.__dict__)
                    if t.solver.changed_from_last:
                        st.write("Changed from last:", t.solver.changed_from_last)

    with tabs[2]:
        st.subheader("RAG & Analytics")
        st.write("DB ready:", db_ready)
        if db_ready:
            st.write("Indexed records:", len(db.records))
        if "retrieved" in st.session_state:
            st.write("Last retrieval set:")
            st.json(st.session_state["retrieved"])
        if "run_results" in st.session_state:
            traces = st.session_state["run_results"].get("traces", [])
            if traces:
                complexity_series = {t.iteration: complexity_to_numeric(t.critic.complexity_class) for t in traces}
                st.line_chart(complexity_series)


def complexity_to_numeric(cls: str) -> int:
    order = {
        "O(1)": 1,
        "O(logN)": 2,
        "O(N)": 3,
        "O(NlogN)": 4,
        "O(N^2)": 5,
        "O(N^3)": 6,
        "O(2^N)": 7,
        "O(N!)": 8,
    }
    return order.get(cls, 0)


def render_guide_markdown(guide: Any) -> str:
    g = guide.__dict__ if hasattr(guide, "__dict__") else guide
    title = g.get("guide_title", "Guide")
    summary = g.get("final_summary", "")
    steps = g.get("steps", []) or []
    pitfalls = g.get("pitfalls", []) or []
    final_complexity = g.get("final_complexity", {}) or {}

    parts = [f"#### {title}", summary or ""]
    if steps:
        parts.append("**Iteration Recap**")
        for step in steps:
            parts.append(
                "- Iteration {iteration}: issue/risk — {issue}. Change — {change}. Evidence — {evidence}. Complexity {before} → {after}.".format(
                    iteration=step.get("iteration", "?"),
                    issue=step.get("what_failed_or_risk", ""),
                    change=step.get("what_we_changed", ""),
                    evidence=step.get("evidence", ""),
                    before=step.get("complexity_before_after", {}).get("before", "unknown"),
                    after=step.get("complexity_before_after", {}).get("after", "unknown"),
                )
            )
    if pitfalls:
        parts.append("**Pitfalls & Reminders**")
        for p in pitfalls:
            parts.append(f"- {p}")
    if final_complexity:
        parts.append(
            "**Final Complexity**: time {time}, space {space}".format(
                time=final_complexity.get("time", "unknown"),
                space=final_complexity.get("space", "unknown"),
            )
        )
    return "\n\n".join(parts)


def render_critic_summary(critic: Any) -> str:
    c = critic.__dict__ if hasattr(critic, "__dict__") else critic
    status = "passed" if c.get("passed") else "failed"
    failure = c.get("failure_type", "?")
    notes = c.get("notes", "")
    suggestion = c.get("suggested_fix", "")
    test_summary = c.get("test_summary", {}) or {}
    num_tests = test_summary.get("num_tests", 0)
    num_passed = test_summary.get("num_passed", 0)
    first_failure = test_summary.get("first_failure")

    human_summary = "The critic {status} the solution (type: {failure}).".format(
        status="approved" if c.get("passed") else "rejected", failure=failure
    )

    lines = [human_summary, f"Tests passed: {num_passed}/{num_tests}."]
    if first_failure:
        lines.append(
            "First failing case #{idx}: expected `{exp}` but got `{got}`.".format(
                idx=first_failure.get("idx"),
                exp=str(first_failure.get("expected", "")),
                got=str(first_failure.get("got", "")),
            )
        )
    if notes:
        lines.append(f"Reasoning: {notes}")
    if suggestion:
        lines.append(f"Suggested next change: {suggestion}")
    lines.append(
        "Complexity gate: {cls}. Evidence: {evidence}".format(
            cls=c.get("complexity_class", "unknown"),
            evidence="; ".join(c.get("complexity_evidence", []) or []),
        )
    )
    return "\n".join(lines)


if __name__ == "__main__":
    main()
