from confluent_kafka import Consumer, Message
import json
import logging
from ner.kafka_functions import SubmittedMessage
from ner.config import (
    DEFAULT_TEXT_PROCESSED_COLLECTION,
    DEFAULT_SENTENCE_SPLIT_COLLECTION,
    KAFKA_TOPIC,
)
from ner.processing import NER_TextProcessor, NerOutputProcessor


class KafkaEventListener:
    def __init__(
        self, bootstrap_servers, group_id, topics, auto_offset_reset="earliest"
    ):
        self.consumer_config = {
            "bootstrap.servers": bootstrap_servers,
            "group.id": group_id,
            "auto.offset.reset": auto_offset_reset,
        }
        self.consumer = Consumer(self.consumer_config)

        self.topics = topics
        self.consumer.subscribe(topics=self.topics)
        self.logger = logging.getLogger(__name__)

        self.text_processor = NER_TextProcessor()
        self.output_processor = NerOutputProcessor()

    def listen(self, timeout: float = 5.0, backoff: float = 5.0):
        while True:
            try:
                data = self.consumer.poll(timeout)

                if data is None:
                    continue
                elif data.error() and not data.error().retriable():
                    raise Exception(
                        f"Error occured with status {data.error().code()}"
                        f"due to reason {data.error()}"
                    )
                else:
                    self.route_data(data)
            except Exception as e:
                self.logger.error(f"Error thrown from {e}")

    def route_data(self, data: Message):
        msg_data = json.loads(data.value().decode("utf-8"))

        submitted_msg = SubmittedMessage(**msg_data)

        if submitted_msg.type == DEFAULT_TEXT_PROCESSED_COLLECTION:
            self.text_processor.run(submitted_msg.data)
        elif submitted_msg.type == DEFAULT_SENTENCE_SPLIT_COLLECTION:
            self.output_processor.run(submitted_msg.data)
        else:
            self.logger.error(
                f"Wrong type {submitted_msg.type} submitted to topic {KAFKA_TOPIC}."
            )
