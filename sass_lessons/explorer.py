#!/usr/bin/env python3
"""
SASS Explorer - Interactive line-by-line SASS assembly explorer
Press Enter to step through actual SASS code with explanations.
"""

import os
import sys
import re
from typing import List

# Import shared components from base
from sass_lessons.base import Colors, SASSExplanations


class SASSInstruction:
    """Represents a single SASS instruction with metadata."""

    def __init__(self, address: str, instruction: str, comment: str = "", source_line: int = 0):
        self.address = address
        self.instruction = instruction
        self.comment = comment
        self.source_line = source_line
        self.mnemonic = self._extract_mnemonic()

    def _extract_mnemonic(self) -> str:
        """Extract the instruction mnemonic."""
        parts = self.instruction.strip().split()
        if parts:
            # Handle predicated instructions
            if parts[0].startswith('@'):
                return parts[1] if len(parts) > 1 else ""
            return parts[0]
        return ""


class SASSExplorer:
    """Interactive SASS code explorer."""

    def __init__(self, sass_file: str):
        self.sass_file = sass_file
        self.instructions: List[SASSInstruction] = []
        self.current_index = 0
        self.explanations = SASSExplanations()
        self._load_sass()

    def _load_sass(self):
        """Load and parse SASS file."""
        if not os.path.exists(self.sass_file):
            print(f"Error: File {self.sass_file} not found!")
            print("Run: SASS_ARCH=sm_80 bash scripts/build.sh")
            sys.exit(1)

        with open(self.sass_file, 'r') as f:
            lines = f.readlines()

        in_kernel = False
        for line in lines:
            # Look for kernel start
            if 'vec_add:' in line and '.text.vec_add:' not in line:
                in_kernel = True
                continue

            if in_kernel:
                # Check for kernel end
                if '.L_x_' in line or line.strip().startswith('.'):
                    if 'EXIT' not in line:
                        break

                # Parse instruction lines
                match = re.match(r'\s*/\*([0-9A-Fa-f]+)\*/\s+(.+)', line)
                if match:
                    addr = match.group(1)
                    inst = match.group(2).strip()

                    # Skip NOP and alignment
                    if inst == 'NOP;' or inst == 'NOP':
                        continue

                    self.instructions.append(SASSInstruction(addr, inst))

                # Parse source comments
                elif '//##' in line:
                    if self.instructions:
                        # Add source reference to previous instruction
                        source_match = re.search(r'line (\d+)', line)
                        if source_match:
                            self.instructions[-1].source_line = int(source_match.group(1))

    def clear_screen(self):
        """Clear terminal screen."""
        os.system('clear' if os.name == 'posix' else 'cls')

    def display_header(self):
        """Display the header."""
        print(f"{Colors.BOLD}{Colors.HEADER}╔══════════════════════════════════════════════════════════╗{Colors.ENDC}")
        print(f"{Colors.BOLD}{Colors.HEADER}║           SASS EXPLORER - LESSON 01: VECTOR ADD          ║{Colors.ENDC}")
        print(f"{Colors.BOLD}{Colors.HEADER}╚══════════════════════════════════════════════════════════╝{Colors.ENDC}")

    def display_instruction(self, inst: SASSInstruction, highlight: bool = True):
        """Display a single instruction with highlighting."""
        # Address
        addr_str = f"{Colors.DIM}[{inst.address}]{Colors.ENDC}"

        # Instruction with syntax highlighting
        if highlight:
            inst_str = self._highlight_instruction(inst.instruction)
        else:
            inst_str = f"{Colors.DIM}{inst.instruction}{Colors.ENDC}"

        # Source line reference
        source_str = ""
        if inst.source_line > 0:
            source_str = f" {Colors.BLUE}// Line {inst.source_line}{Colors.ENDC}"

        print(f"\n{addr_str}  {inst_str}{source_str}")

    def _highlight_instruction(self, instruction: str) -> str:
        """Apply syntax highlighting to instruction."""
        parts = instruction.split(None, 1)
        if not parts:
            return instruction

        # Handle predicated instructions
        if parts[0].startswith('@'):
            pred = f"{Colors.YELLOW}{parts[0]}{Colors.ENDC}"
            if len(parts) > 1:
                rest_parts = parts[1].split(None, 1)
                if rest_parts:
                    mnemonic = f"{Colors.CYAN}{Colors.BOLD}{rest_parts[0]}{Colors.ENDC}"
                    operands = rest_parts[1] if len(rest_parts) > 1 else ""
                    return f"{pred} {mnemonic} {self._highlight_operands(operands)}"
            return pred
        else:
            mnemonic = f"{Colors.CYAN}{Colors.BOLD}{parts[0]}{Colors.ENDC}"
            operands = parts[1] if len(parts) > 1 else ""
            return f"{mnemonic} {self._highlight_operands(operands)}"

    def _highlight_operands(self, operands: str) -> str:
        """Highlight operands in instruction."""
        # Highlight registers
        operands = re.sub(r'\bR(\d+)\b', f'{Colors.GREEN}R\\1{Colors.ENDC}', operands)
        operands = re.sub(r'\bP(\d+)\b', f'{Colors.YELLOW}P\\1{Colors.ENDC}', operands)
        operands = re.sub(r'\bRZ\b', f'{Colors.RED}RZ{Colors.ENDC}', operands)
        operands = re.sub(r'\bPT\b', f'{Colors.YELLOW}PT{Colors.ENDC}', operands)

        # Highlight constants
        operands = re.sub(r'c\[0x[0-9a-fA-F]+\]\[0x[0-9a-fA-F]+\]',
                         lambda m: f'{Colors.BLUE}{m.group()}{Colors.ENDC}', operands)

        # Highlight immediates
        operands = re.sub(r'\b0x[0-9a-fA-F]+\b',
                         lambda m: f'{Colors.YELLOW}{m.group()}{Colors.ENDC}', operands)

        return operands

    def display_explanation(self, inst: SASSInstruction):
        """Display detailed explanation for current instruction."""
        explanation = self.explanations.get_explanation(inst.mnemonic)

        print(f"\n{Colors.BOLD}Instruction: {explanation.get('name', inst.mnemonic)}{Colors.ENDC}")
        print(f"  {Colors.GREEN}→ {explanation.get('desc', '')}{Colors.ENDC}")

        if 'detail' in explanation:
            print(f"\n{Colors.DIM}Details:{Colors.ENDC}")
            for line in explanation['detail'].split('. '):
                if line.strip():
                    print(f"  • {line.strip()}")

        # Special handling for specific instructions
        if inst.mnemonic == "S2R" and "SR_" in inst.instruction:
            # Extract special register
            sr_match = re.search(r'SR_\w+\.\w+|SR_\w+', inst.instruction)
            if sr_match:
                sr_name = sr_match.group()
                special_regs = explanation.get('special_regs', {})
                if sr_name in special_regs:
                    print(f"\n{Colors.YELLOW}Special Register:{Colors.ENDC}")
                    print(f"  {sr_name} = {special_regs[sr_name]}")

        # Show modifiers
        if '.' in inst.mnemonic:
            modifiers = inst.mnemonic.split('.')[1:]
            if modifiers:
                print(f"\n{Colors.DIM}Modifiers:{Colors.ENDC}")
                for mod in modifiers:
                    print(f"  .{mod}")

    def display_context(self):
        """Display context around current instruction."""
        print(f"\n{Colors.BOLD}Context:{Colors.ENDC}")

        # Show previous instruction
        if self.current_index > 0:
            prev = self.instructions[self.current_index - 1]
            print(f"  {Colors.DIM}Previous: {prev.instruction}{Colors.ENDC}")

        # Show next instruction
        if self.current_index < len(self.instructions) - 1:
            next_inst = self.instructions[self.current_index + 1]
            print(f"  {Colors.DIM}Next:     {next_inst.instruction}{Colors.ENDC}")

        # Progress
        progress = f"{self.current_index + 1}/{len(self.instructions)}"
        bar_width = 40
        filled = int(bar_width * (self.current_index + 1) / len(self.instructions))
        bar = '█' * filled + '░' * (bar_width - filled)
        print(f"\n  Progress: [{bar}] {progress}")

    def run(self):
        """Run the interactive explorer."""
        if not self.instructions:
            print("No instructions found in SASS file!")
            return

        print(f"{Colors.BOLD}Welcome to SASS Explorer!{Colors.ENDC}")
        print(f"File: {self.sass_file}")
        print(f"Found {len(self.instructions)} instructions")
        print("\nControls:")
        print("  [ENTER] - Next instruction")
        print("  [b]     - Previous instruction")
        print("  [r]     - Restart from beginning")
        print("  [j]     - Jump to instruction")
        print("  [q]     - Quit")
        print("\nPress [ENTER] to start...")
        input()

        try:
            while self.current_index < len(self.instructions):
                self.clear_screen()
                self.display_header()

                # Display current instruction
                current = self.instructions[self.current_index]
                self.display_instruction(current)
                self.display_explanation(current)
                self.display_context()

                # Get user input
                print(f"\n{Colors.BLUE}[ENTER=next, b=back, r=restart, j=jump, q=quit]: {Colors.ENDC}", end='')
                cmd = input().strip().lower()

                if cmd == 'q':
                    break
                elif cmd == 'b' and self.current_index > 0:
                    self.current_index -= 1
                elif cmd == 'r':
                    self.current_index = 0
                elif cmd == 'j':
                    print("Jump to instruction (1-{}): ".format(len(self.instructions)), end='')
                    try:
                        target = int(input()) - 1
                        if 0 <= target < len(self.instructions):
                            self.current_index = target
                    except ValueError:
                        pass
                else:  # Default: next
                    self.current_index += 1

            if self.current_index >= len(self.instructions):
                print(f"\n{Colors.GREEN}{Colors.BOLD}Completed!{Colors.ENDC}")
                print("You've explored all instructions in this kernel.")

        except KeyboardInterrupt:
            print(f"\n\n{Colors.YELLOW}Explorer interrupted.{Colors.ENDC}")


def main():
    """Main entry point for standalone execution."""
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description='Interactive SASS Assembly Explorer')
    parser.add_argument('sass_file',
                       help='SASS file to explore')

    args = parser.parse_args()

    if not Path(args.sass_file).exists():
        print(f"Error: File {args.sass_file} not found!")
        sys.exit(1)

    explorer = SASSExplorer(args.sass_file)
    explorer.run()


if __name__ == "__main__":
    main()