import pytest
from src.core.gaia import KnowledgeBase, TaskOrchestrator

def test_knowledge_base_add():
    kb = KnowledgeBase()
    kb.add("Test data", "test", {"source": "web"})
    assert len(kb.data) == 1
    assert kb.data[0]["text"] == "Test data"

def test_task_orchestrator_define():
    kb = KnowledgeBase()
    orchestrator = TaskOrchestrator(None, None, kb, None)
    task = orchestrator.define_task("Test task")
    assert task["description"] == "Test task"
    assert len(task["subtasks"]) > 0
