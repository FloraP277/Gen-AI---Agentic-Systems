"""Group the 77 intents into team queues for routing.

Rules are checked in order; the first keyword found in the intent name wins.
Anything unmatched goes to "account". Print the mapping with
`python -m evalharness.queues` and adjust the rules if a group looks wrong.
"""

QUEUE_RULES = [
    ("security", ["compromised", "stolen", "lost", "pin", "passcode", "verify", "identity", "fraud"]),
    ("disputes", ["not_recognised", "charged_twice", "refund", "wrong_amount", "extra_charge", "reverted"]),
    ("cards", ["card", "contactless", "apple_pay", "visa_or_mastercard"]),
    ("transfers", ["transfer", "beneficiary", "receiving_money"]),
    ("top_ups", ["top_up", "topping_up"]),
    ("cash", ["cash", "atm"]),
    ("fx", ["exchange", "currency", "fiat"]),
]
DEFAULT_QUEUE = "account"
# Queues where a person must review before anyone replies to the customer.
HUMAN_REVIEW = {"security", "disputes"}


def queue_for(intent: str) -> str:
    name = intent.lower()
    for queue, words in QUEUE_RULES:
        if any(w in name for w in words):
            return queue
    return DEFAULT_QUEUE


def needs_human(intent: str) -> bool:
    return queue_for(intent) in HUMAN_REVIEW


def all_queues() -> list[str]:
    return [q for q, _ in QUEUE_RULES] + [DEFAULT_QUEUE]


if __name__ == "__main__":
    from .data import load_banking77

    by_queue = {}
    for intent in load_banking77().intents:
        by_queue.setdefault(queue_for(intent), []).append(intent)
    for q in all_queues():
        print(f"{q} ({len(by_queue.get(q, []))}): {', '.join(by_queue.get(q, []))}")
