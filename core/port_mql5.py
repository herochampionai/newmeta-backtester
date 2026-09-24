"""CLI tool to auto-port any MQL5 .mq5 EA to a Python strategy class.

Usage:
    python -m core.port_mql5 "C:\\path\\to\\strategy.mq5" --strategy-name my_strategy

Outputs:
    strategies/<dir>/<strategy_name>_strategy.py

The generated strategy is auto-registered in strategies/__init__.py.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
import re
import os

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from core.mql5_parser import MQL5Parser


def port_mql5(mq5_path: str, strategy_name: str | None = None, registry: bool = True) -> str:
    """Parse an MQL5 EA and generate a Python strategy class file."""
    path = Path(mq5_path)
    if not path.exists():
        raise FileNotFoundError(f"MQL5 file not found: {path}")

    parser = MQL5Parser(path)
    spec = parser.parse()

    if strategy_name is None:
        # Derive from EA name
        ea_name = spec["ea_name"]
        strategy_name = re.sub(r'[^a-zA-Z0-9_]', '_', ea_name).lower().replace('_', '')
        strategy_name = strategy_name.replace('lightprimaryprofessionaledition', 'light9')
        strategy_name = strategy_name.replace('ultratradingsystem', '').rstrip('_')

    # Determine output directory
    strat_dir = ROOT / "strategies" / strategy_name
    strat_dir.mkdir(parents=True, exist_ok=True)

    # Generate the class
    class_code = parser.generate_python_class(strategy_name=strategy_name)

    # Write the strategy file
    output_file = strat_dir / f"{strategy_name}_strategy.py"
    output_file.write_text(class_code)

    print(f"[{strategy_name}] Generated {output_file}")
    print(f"  - {len(spec['inputs'])} input params")
    print(f"  - {len(spec['profiles'])} profiles: {list(spec['profiles'].values())}")
    print(f"  - Preset params: {', '.join(f'{k}={len(v)}' for k,v in spec['profile_presets'].items())}")

    # Create __init__.py for the strategy package
    init_file = strat_dir / "__init__.py"
    # Derive class name consistently with generate_python_class
    if strategy_name:
        class_name = strategy_name.title().replace('_', '') + "Strategy"
    else:
        class_name = re.sub(r'[^a-zA-Z0-9_]', '_', spec["ea_name"]).replace('_', ' ').title().replace(' ', '')
    if not class_name.endswith("Strategy"):
        class_name += "Strategy"

    init_content = f'''from .{strategy_name}_strategy import {class_name}

__all__ = ["{class_name}"]
'''
    init_file.write_text(init_content)

    # Add to strategies/__init__.py registry if requested
    if registry:
        add_to_registry(strategy_name, class_name, spec)

    return str(output_file)


def add_to_registry(strategy_name: str, class_name: str, spec: dict) -> None:
    """Add the generated strategy to the main strategies/__init__.py registry."""
    init_path = ROOT / "strategies" / "__init__.py"
    content = init_path.read_text()

    # Check if already registered
    if f'"{strategy_name}": {class_name}' in content:
        print(f"  - Already registered in strategies/__init__.py")
        return

    # Add import after the last strategy import
    import_line = f"from .{strategy_name}.{strategy_name}_strategy import {class_name}"
    if import_line not in content:
        # Find the last import line in the strategy section
        lines = content.split("\n")
        last_import_idx = 0
        for i, line in enumerate(lines):
            if line.startswith("from .") and "_strategy import" not in line and "PATTERN_VARIANTS" not in line:
                last_import_idx = i
        # Insert after the last strategy import
        lines.insert(last_import_idx + 1, import_line)

        # Add to TRADABLE_STRATEGY_REGISTRY or COMPOSITE_STRATEGY_REGISTRY
        if spec["profiles"]:
            # Has profiles — it's a strategy with presets, add to TRADABLE
            # Find TRADABLE_STRATEGY_REGISTRY
            registry_section = None
            for i, line in enumerate(lines):
                if "TRADABLE_STRATEGY_REGISTRY" in line:
                    registry_section = i
                    break
            if registry_section:
                # Find the closing } of this dict
                for i in range(registry_section, len(lines)):
                    if lines[i].strip() == "}":
                        lines.insert(i, f'    "{strategy_name}": {class_name},')
                        break

        # Also add to __all__
        all_idx = None
        for i, line in enumerate(lines):
            if '__all__' in line:
                all_idx = i
                break
        if all_idx:
            for i in range(all_idx, len(lines)):
                if lines[i].strip() == "]":
                    lines.insert(i, f'    "{class_name}",')
                    break

        init_path.write_text("\n".join(lines))
        print(f"  - Registered in strategies/__init__.py")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Auto-port MQL5 EA to Python strategy")
    ap.add_argument("mq5_file", help="Path to .mq5 EA file")
    ap.add_argument("--strategy-name", "-n", help="Output strategy name (default: auto-derived)")
    ap.add_argument("--no-register", action="store_true", help="Don't add to registry")
    args = ap.parse_args()

    port_mql5(args.mq5_file, args.strategy_name, registry=not args.no_register)
