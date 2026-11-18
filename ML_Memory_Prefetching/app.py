import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from memory_trace_generator import generate_trace, generate_dataset, split_traces
from ml_predictor import RandomForestPredictor
from performance_metrics import compare_configurations, simulate_trace, metric_definitions

st.set_page_config(page_title="ML Memory Prefetching", page_icon="🧠", layout="wide")
st.title("🧠 ML-Based Memory Prefetching System")
st.caption("Cache Memory Module | FIFO · LRU · Random Forest Adaptive Prefetching")

@st.cache_data
def load_or_generate():
    path = Path("data/synthetic_traces.csv")
    if path.exists():
        return pd.read_csv(path)
    return generate_dataset(path)

data = load_or_generate()
train, validation, test = split_traces(data)
with st.sidebar:
    st.header("Simulation settings")
    cache_size = st.selectbox("Cache capacity (blocks)", [2, 4, 8, 16, 32], index=2)
    block_size = st.selectbox("Block size (bytes)", [4, 8, 16, 32, 64], index=0)
    policy = st.selectbox("Replacement policy", ["LRU", "FIFO"])
    threshold = st.slider("ML confidence threshold", 0.0, 1.0, 0.55, 0.05)
    pattern = st.selectbox("Synthetic trace pattern",
                           ["mixed", "sequential", "random", "repeated", "loop"])
    length = st.slider("Trace length", 50, 2000, 500, 50)
    use_ml = st.toggle("Enable Random Forest prefetching", True)

tab1, tab2, tab3, tab4 = st.tabs(["Simulation", "Comparison", "Model evaluation", "Metric definitions"])
with tab1:
    st.subheader("Memory access trace")
    default_trace = "0,1,2,3,4,5,2,3,4,5,6,7,8,9,4,5,6,7"
    raw = st.text_area("Enter block sequence (comma-separated)", default_trace, height=90)
    col_a, col_b = st.columns(2)
    with col_a:
        if st.button("Generate selected synthetic trace"):
            st.session_state["active_trace"] = generate_trace(pattern, length, seed=101, trace_id="dashboard_trace")
    with col_b:
        if st.button("Use held-out test trace"):
            st.session_state["active_trace"] = test[test.trace_id == test.trace_id.iloc[0]].copy()
    trace = st.session_state.get("active_trace")
    if trace is None:
        try:
            blocks = [int(x.strip()) for x in raw.split(",") if x.strip()]
            if not blocks or min(blocks) < 0:
                raise ValueError
            trace = pd.DataFrame({
                "trace_id": "manual", "step": range(len(blocks)), "block": blocks,
                "address": [b * block_size for b in blocks],
                "page_id": [b // 16 for b in blocks], "page_number": [b // 16 for b in blocks],
                "page_size": 16 * block_size, "segment_id": [b % 3 for b in blocks],
                "segment_type": [["Code", "Data", "Stack"][b % 3] for b in blocks],
                "segment_offset": [(b % 16) * block_size for b in blocks]
            })
        except Exception:
            st.error("Enter a valid comma-separated sequence of non-negative integers.")
            st.stop()
    if st.button("Run cache simulation", type="primary"):
        try:
            if "active_trace" not in st.session_state:
                blocks = [int(x.strip()) for x in raw.split(",") if x.strip()]
                if not blocks or min(blocks) < 0:
                    raise ValueError
                trace = pd.DataFrame({
                    "trace_id": "manual", "step": range(len(blocks)), "block": blocks,
                    "address": [b * block_size for b in blocks],
                    "page_id": [b // 16 for b in blocks], "page_number": [b // 16 for b in blocks],
                    "page_size": 16 * block_size, "segment_id": [b % 3 for b in blocks],
                    "segment_type": [["Code", "Data", "Stack"][b % 3] for b in blocks],
                    "segment_offset": [(b % 16) * block_size for b in blocks]
                })
            predictor = RandomForestPredictor().fit(train) if use_ml else None
            mode = "ml" if use_ml else "normal"
            st.session_state["last_result"] = simulate_trace(
                trace, cache_size, block_size, policy, mode, predictor, threshold)
        except (ValueError, RuntimeError) as exc:
            st.error(str(exc))
    if "last_result" in st.session_state:
        r = st.session_state["last_result"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Cache hits", r["hits"])
        c2.metric("Cache misses", r["misses"])
        c3.metric("Hit ratio", f'{r["hit_ratio"]:.1%}')
        c4.metric("Miss ratio", f'{r["miss_ratio"]:.1%}')
        st.write("**Final cache contents**")
        st.dataframe(pd.DataFrame(r["cache_contents"]), use_container_width=True)
        st.json({k: v for k, v in r.items() if k != "cache_contents"})

with tab2:
    st.subheader("Fair comparison on the same trace")
    selected = test[test.trace_id == test.trace_id.iloc[0]].copy()
    if st.button("Run all configurations with FIFO and LRU"):
        model = RandomForestPredictor().fit(train)
        frames = []
        for replacement in ["FIFO", "LRU"]:
            frames.append(compare_configurations(
                selected, cache_size, block_size, replacement, model, threshold))
        st.session_state["comparison"] = pd.concat(frames, ignore_index=True)
    if "comparison" in st.session_state:
        comp = st.session_state["comparison"]
        display_cols = ["mode", "policy", "hits", "misses", "hit_ratio",
                        "miss_ratio", "prefetch_accuracy", "prefetch_coverage",
                        "cache_pollution", "average_memory_access_time"]
        st.dataframe(comp[display_cols], use_container_width=True)
        pivot = comp.pivot(index="mode", columns="policy", values="hit_ratio")
        fig, ax = plt.subplots()
        pivot.plot(kind="bar", ax=ax)
        ax.set_ylabel("Hit ratio")
        ax.set_title("Hit ratio: FIFO vs LRU")
        ax.set_ylim(0, 1)
        st.pyplot(fig)
        pivot_time = comp.pivot(index="mode", columns="policy",
                                values="average_memory_access_time")
        fig2, ax2 = plt.subplots()
        pivot_time.plot(kind="bar", ax=ax2)
        ax2.set_ylabel("Average access time (latency units)")
        ax2.set_title("Average memory access time: FIFO vs LRU")
        st.pyplot(fig2)

with tab3:
    st.subheader("Random Forest evaluation on unseen traces")
    model = RandomForestPredictor().fit(train)
    val_metrics = model.evaluate(validation)
    test_metrics = model.evaluate(test)
    st.write("The model is trained on complete training traces and evaluated on disjoint validation/test traces.")
    st.columns(2)[0].json({"Validation": val_metrics})
    st.columns(2)[1].json({"Held-out test": test_metrics})
    st.write("Training traces:", train.trace_id.nunique(),
             "| Validation traces:", validation.trace_id.nunique(),
             "| Test traces:", test.trace_id.nunique())

with tab4:
    st.subheader("Metric definitions")
    for name, formula in metric_definitions().items():
        st.markdown(f"**{name}:** {formula}")
    st.info("Latency is a configurable simulation cost model, not measured hardware latency. Default values: cache=1 and main memory=100 latency units.")

st.divider()
st.caption("Research note: synthetic traces are reproducible with fixed seeds. Replace them with imported traces using the documented CSV schema.")
