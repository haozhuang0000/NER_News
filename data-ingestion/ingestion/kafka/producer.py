from dataclasses import dataclass, asdict
import json
from confluent_kafka import Producer
from ingestion.models import Article
from ingestion.config import BOOTSTRAP_SERVERS


@dataclass
class SubmittedMessage:
    type: str
    data: list[Article]


def submit_data(
    messages: list[Article],
    bootstrap_servers=BOOTSTRAP_SERVERS,
    topic_name: str = "news",
):
    producer = Producer(
        {
            "bootstrap.servers": bootstrap_servers,
        }
    )
    data = SubmittedMessage(type=topic_name, data=[asdict(msg) for msg in messages])

    producer.produce(topic=topic_name, value=json.dumps(data))
    producer.flush()
