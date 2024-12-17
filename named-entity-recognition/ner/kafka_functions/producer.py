from dataclasses import dataclass, asdict
import json
from confluent_kafka import Producer
from typing import TypeVar
from ner.models import SentenceSplit, NEROut
from ner.config import BOOTSTRAP_SERVERS


T = TypeVar("A", SentenceSplit, NEROut)


@dataclass
class SubmittedMessage:
    type: str
    data: list[SentenceSplit | NEROut]


def submit_data(
    messages: list[SentenceSplit | NEROut],
    bootstrap_servers=BOOTSTRAP_SERVERS,
    topic_name: str = "ner_out",
):
    producer = Producer(
        {
            "bootstrap.servers": bootstrap_servers,
        }
    )
    data = SubmittedMessage(type=topic_name, data=[asdict(msg) for msg in messages])

    producer.produce(topic=topic_name, value=json.dumps(data))
    producer.flush()
