import os
import time

CURRENT_LOCAL_RANK = os.getenv("LOCAL_RANK", "UNKNOWN")
CURRENT_RANK = os.getenv("RANK", "UNKNOWN")
CURRENT_WS = os.getenv("WORLD_SIZE", "UNKNOWN")

print(f"Hello from training script! LR:({CURRENT_LOCAL_RANK}) R:({CURRENT_RANK}) WS:({CURRENT_WS})")

for i in range(30):
    print(f"LR:({CURRENT_LOCAL_RANK}) | {i}")
    time.sleep(1)
