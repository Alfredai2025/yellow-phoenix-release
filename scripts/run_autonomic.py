#!/usr/bin/env python3
"""
🧪 Yellow Phoenix Autonomic Phase 4 Test
Runs SIE loop with 4 real sensors + 3 real actuators for 2 minutes.
"""
import sys
import time
import logging
from pathlib import Path

BASE = Path(__file__).parent.parent
sys.path.insert(0, str(BASE))

from yp_autonomic.agent import AutonomicAgent
from yp_autonomic.sensors.system_hygiene import SystemHygieneSensor
from yp_autonomic.sensors.database_health import DatabaseHealthSensor
from yp_autonomic.sensors.service_health import ServiceHealthSensor
from yp_autonomic.sensors.resource_pressure import ResourcePressureSensor
from yp_autonomic.actuators import ProcessActuator, DatabaseActuator, NotifyActuator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)-30s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)


def phase4_test():
    print("\n" + "=" * 60)
    print("🧪 YP AUTONOMIC PHASE 4 — REAL ACTUATORS")
    print("=" * 60 + "\n")

    sensors = [
        SystemHygieneSensor(),
        DatabaseHealthSensor(),
        ServiceHealthSensor(),
        ResourcePressureSensor(),
    ]
    actuators = {
        "process": ProcessActuator(),
        "database": DatabaseActuator(),
        "notify": NotifyActuator(),
    }

    agent = AutonomicAgent(sensors=sensors, actuators=actuators)

    def stop_after():
        time.sleep(120)
        print("\n🛑 Stopping agent...")
        agent.stop()

    stopper = __import__('threading').Thread(target=stop_after, daemon=True)
    stopper.start()

    try:
        agent.run()
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
        agent.stop()

    print("\n" + "=" * 60)
    print("📊 FINAL STATUS")
    print("=" * 60)
    status = agent.status()
    for k, v in status.items():
        print(f"  {k}: {v}")

    inbox = BASE / "data" / "operator_inbox.txt"
    if inbox.exists():
        lines = inbox.read_text().strip().split("\n")
        print(f"\n  📨 Operator inbox: {len(lines)} notification(s)")
        for line in lines[-5:]:
            print(f"    {line}")

    print("\n✅ PHASE 4 TEST COMPLETE")
    return 0


if __name__ == "__main__":
    sys.exit(phase4_test())
