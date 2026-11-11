"""
ממשק Streamlit להדגמת Human in the Loop.

אותו גרף בדיוק מ-main.py, רק עם מסך שמראה:
- איפה הגרף עומד עכשיו (הצומת המודגש בתרשים)
- מה יש ב-state
- אילו nodes רצו בכל הרצה

הרצה:
    streamlit run app.py
"""

# =========================
# CONFIG
# =========================
import uuid

import streamlit as st
from langgraph.types import Command

from main import build_graph

st.set_page_config(page_title="Human in the Loop", page_icon="🧑‍💻", layout="wide")

# יישור לימין לטקסט בעברית
st.markdown(
    """
    <style>
    .stApp, .stMarkdown, .stTextArea textarea, .stTextInput input { direction: rtl; text-align: right; }
    code, pre { direction: ltr; text-align: left; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_graph():
    """
    Streamlit מריצה את כל הסקריפט מחדש בכל לחיצה.
    cache_resource שומר את הגרף (ואיתו את ה-checkpointer) בין ההרצות,
    אחרת היינו מאבדות את המצב בכל לחיצה.
    """
    return build_graph()


graph = get_graph()

# כל ה-nodes בגרף, לצורך הלוג
NODES: list[str] = ["write_draft", "human_review", "send_email"]


# =========================
# SESSION STATE
# =========================
if "thread_id" not in st.session_state:
    st.session_state.thread_id = None
    st.session_state.log = []  # רשימת שורות: מה רץ בכל הרצה


def config() -> dict:
    return {"configurable": {"thread_id": st.session_state.thread_id}}


# =========================
# HELPERS
# =========================
def run_graph(graph_input) -> None:
    """
    מריצים עם stream במקום invoke, כדי לראות כל node שסיים.
    stream_mode="updates" מחזיר אחרי כל node רק את מה שהוא עדכן.
    """
    for chunk in graph.stream(graph_input, config(), stream_mode="updates"):
        for node_name, update in chunk.items():
            if node_name == "__interrupt__":
                st.session_state.log.append("⏸️ interrupt — הגרף עצר ומחכה לאדם")
            else:
                keys = ", ".join(update.keys()) if update else "—"
                st.session_state.log.append(f"✅ {node_name} רץ (עדכן: {keys})")


def resume(decision: dict) -> None:
    st.session_state.log.append(f"▶️ resume עם {decision}")
    run_graph(Command(resume=decision))


def flow_chart(current: str | None, done: bool) -> str:
    """בונה תרשים DOT של הגרף, עם הצומת הנוכחי מודגש."""

    def node(name: str, label: str) -> str:
        if name == current:
            style = 'style=filled fillcolor="#ffd54f" penwidth=3'
        else:
            style = 'style=filled fillcolor="#eeeeee"'
        return f'{name} [label="{label}" shape=box {style}];'

    end_style = 'style=filled fillcolor="#a5d6a7" penwidth=3' if done else ""
    return f"""
    digraph G {{
        rankdir=TB;
        node [fontname="Arial"];
        START [shape=circle label="START"];
        END [shape=doublecircle label="END" {end_style}];
    n_write = node("write_draft", "write_draft\\nהמודל כותב טיוטה")
    n_review = node("human_review", "human_review\\n⏸ interrupt")
    n_send = node("send_email", "send_email\\nשליחה ללקוח")
        {n_write}
        {n_review}
        {n_send}
        START -> write_draft;
        write_draft -> human_review;
        human_review -> send_email [label="approve / edit"];
        human_review -> write_draft [label="revise" style=dashed];
        human_review -> END [label="cancel"];
        send_email -> END;
    }}
    """


# =========================
# MAIN
# =========================
st.title("🧑‍💻 Human in the Loop עם LangGraph")

left, right = st.columns([3, 2], gap="large")

snapshot = graph.get_state(config()) if st.session_state.thread_id else None
state: dict = snapshot.values if snapshot else {}
waiting_for_human: bool = bool(snapshot and snapshot.interrupts)
finished: bool = bool(snapshot and not snapshot.next and state)

# ---------- צד שמאל: האינטראקציה ----------
with left:
    st.subheader("1. הודעת הלקוח")
    customer_message = st.text_area(
        "מה הלקוח כתב",
        value="הזמנתי לפני שבוע ועדיין לא קיבלתי את החבילה. מה קורה?",
        disabled=st.session_state.thread_id is not None,
    )

    if st.session_state.thread_id is None:
        if st.button("🚀 הפעלת הגרף", type="primary"):
            # thread_id חדש = שיחה חדשה עם state נקי
            st.session_state.thread_id = f"ticket-{uuid.uuid4().hex[:6]}"
            st.session_state.log = []
            with st.spinner("המודל כותב טיוטה..."):
                run_graph({"customer_message": customer_message, "feedback": ""})
            st.rerun()

    if waiting_for_human:
        payload = snapshot.interrupts[0].value
        st.subheader("2. ⏸️ הגרף עצר — ההחלטה שלך")
        st.info(payload["question"])

        edited = st.text_area("הטיוטה (אפשר לערוך)", value=payload["draft"], height=150)
        feedback = st.text_input("הערה למודל (לתיקון)", placeholder="למשל: קצר יותר, ותציעי פיצוי")

        c1, c2, c3, c4 = st.columns(4)
        if c1.button("✅ אישור", use_container_width=True):
            resume({"action": "approve"})
            st.rerun()
        if c2.button("✏️ שליחת הנוסח הערוך", use_container_width=True):
            resume({"action": "edit", "text": edited})
            st.rerun()
        if c3.button("🔁 תיקון ע\"י המודל", use_container_width=True, disabled=not feedback):
            with st.spinner("המודל מתקן..."):
                resume({"action": "revise", "text": feedback})
            st.rerun()
        if c4.button("🛑 ביטול", use_container_width=True):
            resume({"action": "cancel"})
            st.rerun()

    if finished:
        st.subheader("3. סיום")
        if state.get("status") == "sent":
            st.success("📧 נשלח ללקוח:")
            st.write(state["draft"])
        else:
            st.warning("🛑 בוטל. שום דבר לא נשלח.")

    if st.session_state.thread_id and st.button("🔄 התחלה מחדש"):
        st.session_state.thread_id = None
        st.rerun()

# ---------- צד ימין: מה קורה מאחורי הקלעים ----------
with right:
    st.subheader("🗺️ איפה הגרף עומד")
    current = snapshot.next[0] if snapshot and snapshot.next else None
    st.graphviz_chart(flow_chart(current, finished))

    if st.session_state.thread_id:
        st.caption(f"thread_id: `{st.session_state.thread_id}`")

    st.subheader("📜 מה רץ")
    for line in st.session_state.log:
        st.text(line)

    st.subheader("🧠 State")
    st.json(state or {}, expanded=True)

    if st.session_state.thread_id:
        history = list(graph.get_state_history(config()))
        st.caption(f"ה-checkpointer שמר עד עכשיו {len(history)} checkpoints לשיחה הזו")
