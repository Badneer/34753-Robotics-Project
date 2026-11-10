import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from Part1.Problem1.main import T04

o04 = T04[:3, 3]  # Stylus position
x04 = T04[:3, 0]  # Stylus x-axis orientation

if __name__ == "__main__":
    print(f"Stylus position: \n {o04}")
    print(f"Stylus x-axis orientation: \n {x04}")