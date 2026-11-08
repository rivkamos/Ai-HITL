"""
בדיקות לגרף בלי לקרוא ל-OpenAI: מחליפים את המודל במודל מזויף.
הרצה: pytest
"""
import os
import sys
from pathlib import Path

# ChatOpenAI דורש מפתח כבר ביצירה, גם אם לא קוראים לו בפועל
os.environ.setdefault("OPENAI_API_KEY", "test-key")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.output_parsers import StrOutputParser
from langgraph.types import Command

import main


@pytest.fixture
def graph():
    fake_llm = FakeListChatModel(responses=["טיוטה ראשונה", "טיוטה מתוקנת"])
    main.draft_chain = main.DRAFT_PROMPT | fake_llm | StrOutputParser()
    return main.build_graph()


def start(graph, thread_id: str):
    config = {"configurable": {"thread_id": thread_id}}
    result = graph.invoke({"customer_message": "איפה החבילה?", "feedback": ""}, config)
    return config, result


def test_graph_stops_before_sending(graph):
    _, result = start(graph, "t1")
    assert "__interrupt__" in result
    assert result["__interrupt__"][0].value["draft"] == "טיוטה ראשונה"


def test_approve_sends(graph):
    config, _ = start(graph, "t2")
    result = graph.invoke(Command(resume={"action": "approve"}), config)
    assert result["status"] == "sent"


def test_edit_uses_human_text(graph):
    config, _ = start(graph, "t3")
    result = graph.invoke(Command(resume={"action": "edit", "text": "נוסח שלי"}), config)
    assert result["draft"] == "נוסח שלי"
    assert result["status"] == "sent"


def test_revise_goes_back_to_model(graph):
    config, _ = start(graph, "t4")
    result = graph.invoke(Command(resume={"action": "revise", "text": "קצר יותר"}), config)
    # המודל כתב טיוטה חדשה, והגרף עצר שוב לאישור
    assert result["__interrupt__"][0].value["draft"] == "טיוטה מתוקנת"


def test_cancel_does_not_send(graph):
    config, _ = start(graph, "t5")
    result = graph.invoke(Command(resume={"action": "cancel"}), config)
    assert result["status"] == "cancelled"
