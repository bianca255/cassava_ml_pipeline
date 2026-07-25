"""
locustfile.py
-------------
Load test for the Cassava Leaf Disease Classification API.

Usage:
    locust -f locustfile.py --host=http://localhost:8000

Then open http://localhost:8089 to configure number of users and spawn rate.

For the rubric's "flood of requests" requirement, run this against the API
with varying numbers of Docker containers (e.g. `docker compose up --scale
api=1/2/4`) behind a load balancer, and record latency/RPS from the Locust
web UI or CSV export (--csv=results/run1) for each container count.
"""

import os
import random

from locust import HttpUser, task, between

SAMPLE_IMAGE_DIR = os.environ.get("LOCUST_SAMPLE_DIR", "data/test")


def _pick_sample_image():
    """Grabs a random real image from the test set so requests are realistic."""
    for root, _, files in os.walk(SAMPLE_IMAGE_DIR):
        images = [f for f in files if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        if images:
            return os.path.join(root, random.choice(images))
    return None


class CassavaAPIUser(HttpUser):
    wait_time = between(0.5, 2.0)

    def on_start(self):
        self.sample_image = _pick_sample_image()

    @task(5)
    def predict(self):
        if not self.sample_image:
            return
        with open(self.sample_image, "rb") as f:
            self.client.post(
                "/predict",
                files={"file": ("leaf.jpg", f, "image/jpeg")},
                name="/predict",
            )

    @task(2)
    def uptime(self):
        self.client.get("/uptime", name="/uptime")

    @task(1)
    def metrics(self):
        self.client.get("/metrics", name="/metrics")

    @task(1)
    def class_distribution(self):
        self.client.get("/visualizations/class-distribution", name="/visualizations/class-distribution")
