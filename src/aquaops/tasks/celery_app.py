from celery import Celery


celery_app = Celery("aquaops", include=["aquaops.tasks.jobs"])
celery_app.conf.update(
    broker_url="redis://redis:6379/0",
    result_backend="redis://redis:6379/1",
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
)
