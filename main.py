"""
Human in the Loop עם LangGraph
==============================

סוכן שכותב טיוטת תשובה ללקוח, ועוצר לפני השליחה כדי שאדם יאשר.
האדם יכול: לאשר, לערוך בעצמו, לבקש תיקון מהמודל, או לבטל.

הרצה:
    python main.py
"""

# =========================
# CONFIG
# =========================
from typing import Literal, TypedDict

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

# המפתח (OPENAI_API_KEY) נשמר ב-.env ולא נכנס ל-git
load_dotenv()

# מה שנכון היום יכול להיות לא נכון מחר, כדאי להמשיך להתעדכן בשמות המודלים
MODEL_NAME: str = "gpt-6-luna"

# ה-ChatOpenAI קורא את המפתח מהסביבה בעצמו, ולכן לא מעבירים אותו כאן
llm = ChatOpenAI(model=MODEL_NAME, temperature=0)


# =========================
# STATE
# =========================
class EmailState(TypedDict):
    """הזיכרון המשותף של הגרף. כל node קורא ממנו ומחזיר עדכון חלקי."""

    customer_message: str  # מה הלקוח כתב
    draft: str  # הטיוטה הנוכחית
    feedback: str  # הערות של האדם לתיקון (אם יש)
    status: str  # sent / cancelled


# =========================
# PROMPTS
# =========================
# הפרומפט מוגדר פעם אחת כקבוע, ולא נבנה בתוך ה-node
DRAFT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "את נציגת שירות לקוחות מנומסת ותמציתית. "
            "כתבי תשובה קצרה בעברית, עד 4 משפטים.",
        ),
        (
            "human",
            "הודעת הלקוח:\n{customer_message}\n\n"
            "הערות לתיקון מהטיוטה הקודמת (אם ריק - אין):\n{feedback}",
        ),
    ]
)

# LCEL: פרומפט -> מודל -> טקסט נקי. זה החלק של LangChain
draft_chain = DRAFT_PROMPT | llm | StrOutputParser()


# =========================
# NODES
# =========================
def write_draft(state: EmailState) -> dict:
    """המודל כותב טיוטה (או מתקן אותה לפי ה-feedback)."""
    draft = draft_chain.invoke(
        {
            "customer_message": state["customer_message"],
            "feedback": state.get("feedback", ""),
        }
    )
    return {"draft": draft}


def human_review(
    state: EmailState,
) -> Command[Literal["write_draft", "send_email", "__end__"]]:
    """
    כאן הגרף עוצר ומחכה לאדם.

    interrupt() שומר את המצב ב-checkpointer ומחזיר שליטה לקוד שהריץ את הגרף.
    כשממשיכים עם Command(resume=...), הערך שנשלח חוזר כאן כתוצאה של interrupt().

    שימי לב: בהמשכה ה-node הזה רץ שוב מההתחלה. לכן לא שמים לפני interrupt()
    פעולות עם תופעות לוואי (שליחת מייל, כתיבה ל-DB).
    """
    decision: dict = interrupt(
        {
            "draft": state["draft"],
            "question": "לאשר, לערוך, לבקש תיקון או לבטל?",
        }
    )

    action = decision["action"]

    if action == "approve":
        return Command(goto="send_email")

    if action == "edit":
        # האדם כתב את הנוסח הסופי בעצמו
        return Command(update={"draft": decision["text"]}, goto="send_email")

    if action == "revise":
        # חוזרים למודל עם הערות - זו הלולאה של Human in the Loop
        return Command(update={"feedback": decision["text"]}, goto="write_draft")

    return Command(update={"status": "cancelled"}, goto=END)


def send_email(state: EmailState) -> dict:
    """הפעולה ה"מסוכנת". מגיעים אליה רק אחרי אישור אנושי."""
    print("\n📧 נשלח ללקוח:\n" + state["draft"])
    return {"status": "sent"}


# =========================
# GRAPH
# =========================
def build_graph():
    builder = StateGraph(EmailState)

    builder.add_node("write_draft", write_draft)
    builder.add_node("human_review", human_review)
    builder.add_node("send_email", send_email)

    builder.add_edge(START, "write_draft")
    builder.add_edge("write_draft", "human_review")
    # מ-human_review אין edge קבוע: ה-Command שהוא מחזיר מחליט לאן ממשיכים
    builder.add_edge("send_email", END)

    # בלי checkpointer אין interrupt: אין איפה לשמור את המצב בזמן שמחכים לאדם
    return builder.compile(checkpointer=InMemorySaver())


# =========================
# HELPERS
# =========================
def ask_human(payload: dict) -> dict:
    """מציג לאדם את הטיוטה ומחזיר את ההחלטה שלו במבנה ש-human_review מצפה לו."""
    print("\n" + "=" * 50)
    print("📝 טיוטה:\n" + payload["draft"])
    print("=" * 50)
    print(payload["question"])
    choice = input("[a] אישור  [e] עריכה  [r] תיקון ע\"י המודל  [c] ביטול: ").strip().lower()

    if choice == "a":
        return {"action": "approve"}
    if choice == "e":
        return {"action": "edit", "text": input("הנוסח הסופי: ")}
    if choice == "r":
        return {"action": "revise", "text": input("מה לתקן? ")}
    return {"action": "cancel"}


# =========================
# MAIN
# =========================
def main() -> None:
    graph = build_graph()

    # thread_id מזהה את השיחה. לפיו ה-checkpointer יודע מאיפה להמשיך
    config = {"configurable": {"thread_id": "ticket-1"}}

    customer_message = input("הודעת הלקוח: ") or "הזמנתי לפני שבוע ועדיין לא קיבלתי את החבילה. מה קורה?"

    # הרצה ראשונה: הגרף רץ עד ה-interrupt הראשון
    result = graph.invoke({"customer_message": customer_message, "feedback": ""}, config)

    # כל עוד הגרף עצר ומחכה לאדם - שואלים וממשיכים
    while "__interrupt__" in result:
        payload = result["__interrupt__"][0].value
        decision = ask_human(payload)
        result = graph.invoke(Command(resume=decision), config)

    print(f"\n✅ סטטוס סופי: {result.get('status')}")


if __name__ == "__main__":
    main()
