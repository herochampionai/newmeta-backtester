"""Comprehensive Streamlit app test — exercise every mode end-to-end."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from streamlit.testing.v1 import AppTest

print("=== Streamlit App Smoke Test ===\n")

# Test 1: Initial load (default = Backtest mode, no file uploaded)
print("Test 1: Initial load (Backtest mode, no file)")
at = AppTest.from_file("frontend/app.py", default_timeout=60).run()
print(f"  Exceptions: {len(at.exception)}")
print(f"  Errors: {len(at.error)}")
print(f"  Main sections: {len(at.main)}")
if at.exception:
    for e in at.exception:
        print(f"  EXC: {e.value}")
if at.error:
    for e in at.error:
        print(f"  ERR: {e.value}")

# Test 2: Simulate selecting a strategy from the dropdown
print("\nTest 2: Switch to 'fbb' strategy in sidebar")
at.sidebar.selectbox[0].set_value("D:/MT5_EuroPrinter/terminal64.exe").run()
print(f"  Exceptions: {len(at.exception)}")
print(f"  Errors: {len(at.error)}")

# Test 3: Click run button (backtest mode)
print("\nTest 3: Click 'Run Backtest' button")
try:
    # Find the run button
    buttons = [b for b in at.button if "Run" in str(b.label) or "Run" in str(b.key)]
    if buttons:
        buttons[0].click().run()
        print(f"  Exceptions: {len(at.exception)}")
        print(f"  Errors: {len(at.error)}")
    else:
        print("  No run button found")
except Exception as e:
    print(f"  Button click failed: {e}")

# Test 4: Switch to Optimize mode
print("\nTest 4: Switch to Optimize mode")
try:
    radio_buttons = at.sidebar.radio
    for rb in radio_buttons:
        if "Optimize" in str(rb):
            rb.set_value("🔬 Optimize").run()
            print(f"  Exceptions: {len(at.exception)}")
            print(f"  Errors: {len(at.error)}")
            break
    else:
        print("  No optimize radio found")
except Exception as e:
    print(f"  Radio switch failed: {e}")

# Test 5: Switch to Multi-Strategy mode
print("\nTest 5: Switch to Multi-Strategy mode")
try:
    for rb in at.sidebar.radio:
        if "Multi" in str(rb):
            rb.set_value("🧬 Multi-Strategy").run()
            print(f"  Exceptions: {len(at.exception)}")
            print(f"  Errors: {len(at.error)}")
            break
except Exception as e:
    print(f"  Failed: {e}")

# Test 6: Switch to Monte Carlo
print("\nTest 6: Switch to Monte Carlo mode")
try:
    for rb in at.sidebar.radio:
        if "Monte" in str(rb):
            rb.set_value("🎲 Monte Carlo").run()
            print(f"  Exceptions: {len(at.exception)}")
            print(f"  Errors: {len(at.error)}")
            break
except Exception as e:
    print(f"  Failed: {e}")

# Test 7: Switch to Walk-Forward
print("\nTest 7: Switch to Walk-Forward mode")
try:
    for rb in at.sidebar.radio:
        if "Walk" in str(rb):
            rb.set_value("📊 Walk-Forward").run()
            print(f"  Exceptions: {len(at.exception)}")
            print(f"  Errors: {len(at.error)}")
            break
except Exception as e:
    print(f"  Failed: {e}")

print("\n=== All tests complete ===")