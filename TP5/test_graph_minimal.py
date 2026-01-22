# TP5/test_graph_minimal.py
import uuid

from TP5.load_test_emails import load_all_emails
from TP5.agent.state import AgentState
from TP5.agent.graph_minimal import build_graph


if __name__ == "__main__":
    emails = load_all_emails()
    app = build_graph()

    # Sélectionner E05 (reply avec evidence) et E04 (evidence vide/safe mode)
    #email_ids = ["E05", "E04"]
    email_ids = ["E11"]
    test_emails = [e for e in emails if e["email_id"] in email_ids]

    for e in test_emails:
        state = AgentState(
            run_id=str(uuid.uuid4()),
            email_id=e["email_id"],
            subject=e["subject"],
            sender=e["from"],
            body=e["body"],
        )

        out = app.invoke(state)

        print("=" * 50)
        print(f"EMAIL {e['email_id']}")
        print("=== DECISION ===")
        print(out["decision"].model_dump_json(indent=2))
        print("\n=== DRAFT_V1 ===")
        print(out["draft_v1"])
        print("\n=== ACTIONS ===")
        print(out["actions"])

        print("\n=== FINAL ===")
        print("kind =", out["final_kind"])
        print(out["final_text"])
