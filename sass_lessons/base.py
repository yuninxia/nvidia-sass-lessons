#!/usr/bin/env python3
"""
Base classes for SASS lessons - provides extensible framework
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Tuple
import os
import sys
import shutil


# ANSI color codes for terminal
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    DIM = '\033[2m'
    BG_YELLOW = '\033[43m'
    BG_RESET = '\033[49m'


class BaseTutorial(ABC):
    """Base class for interactive tutorials."""

    def __init__(self, lesson_num: int = 1):
        self.lesson_num = lesson_num
        self.step = 0
        self.lessons = self._create_lessons()

    @abstractmethod
    def _create_lessons(self) -> List[Tuple[str, callable]]:
        """Create the lesson sequence. Must be implemented by each lesson."""
        pass

    @abstractmethod
    def get_title(self) -> str:
        """Get the lesson title."""
        pass

    @abstractmethod
    def get_description(self) -> str:
        """Get the lesson description."""
        pass

    def clear_screen(self):
        """Clear the terminal screen."""
        os.system('clear' if os.name == 'posix' else 'cls')

    def print_header(self, text: str):
        """Print a section header."""
        print(f"\n{Colors.BOLD}{Colors.HEADER}{'='*60}{Colors.ENDC}")
        print(f"{Colors.BOLD}{Colors.HEADER}{text.center(60)}{Colors.ENDC}")
        print(f"{Colors.BOLD}{Colors.HEADER}{'='*60}{Colors.ENDC}\n")

    def print_instruction(self, inst: str, comment: str = ""):
        """Print a SASS instruction with syntax highlighting."""
        parts = inst.split()
        if parts:
            colored_inst = f"{Colors.CYAN}{parts[0]}{Colors.ENDC}"
            if len(parts) > 1:
                colored_inst += " " + " ".join(parts[1:])

            print(f"  {Colors.BOLD}{colored_inst}{Colors.ENDC}", end="")
            if comment:
                print(f"  {Colors.DIM}# {comment}{Colors.ENDC}")
            else:
                print()

    def print_explanation(self, text: str, indent: int = 2):
        """Print explanation text with formatting."""
        prefix = " " * indent
        for line in text.strip().split('\n'):
            if line.strip():
                print(f"{prefix}{Colors.GREEN}→ {line}{Colors.ENDC}")

    def print_code_block(self, code: str, lang: str = "", width: int = None):
        """Print a code block with syntax highlighting.

        Args:
            code: The code to display
            lang: Language label for the header
            width: Width of the code block (None = auto, based on terminal width)
        """
        # Get terminal width if not specified
        if width is None:
            try:
                terminal_width = shutil.get_terminal_size().columns
                # Leave some margin for better readability
                width = min(terminal_width - 10, 100)  # Max 100 chars wide
            except:
                width = 70  # Default fallback

        # Calculate the actual box width
        box_width = max(width, max(len(line) for line in code.strip().split('\n')) + 2)

        # Header
        header = f"[{lang}]" if lang else ""
        header_padding = box_width - len(header) - 1
        print(f"\n{Colors.YELLOW}┌─{header}{'─' * header_padding}┐{Colors.ENDC}")

        # Code lines
        for line in code.strip().split('\n'):
            padding = box_width - len(line) - 2
            print(f"{Colors.YELLOW}│{Colors.ENDC} {line}{' ' * padding} {Colors.YELLOW}│{Colors.ENDC}")

        # Footer
        print(f"{Colors.YELLOW}└{'─' * box_width}┘{Colors.ENDC}\n")

    def wait_for_enter(self, prompt: str = ""):
        """Wait for user to press Enter."""
        if prompt:
            input(f"\n{Colors.BLUE}{prompt}{Colors.ENDC}")
        else:
            input(f"\n{Colors.BLUE}Press [ENTER] to continue...{Colors.ENDC}")

    def intro(self):
        """Default introduction."""
        self.clear_screen()
        self.print_header(f"LESSON {self.lesson_num:02d}: {self.get_title()}")
        print(f"{Colors.BOLD}Welcome to Interactive SASS Learning!{Colors.ENDC}\n")
        print(self.get_description())
        print(f"\n{Colors.YELLOW}Instructions:{Colors.ENDC}")
        print("• Press [ENTER] to advance to the next step")
        print("• Press Ctrl+C to exit at any time")
        print("• Each step builds on the previous one\n")

    def summary(self):
        """Default summary."""
        self.clear_screen()
        self.print_header("LESSON SUMMARY")

        print(f"{Colors.GREEN}Congratulations! You've completed Lesson {self.lesson_num:02d}!{Colors.ENDC}\n")

        # Show what was learned (can be overridden)
        self.show_key_takeaways()

        if self.lesson_num < 9:
            print(f"\nNext: Lesson {self.lesson_num + 1:02d}")

    def show_key_takeaways(self):
        """Show key takeaways. Can be overridden by each lesson."""
        print(f"{Colors.BOLD}Key Takeaways:{Colors.ENDC}\n")
        print("• You've learned about GPU assembly")
        print("• SASS instructions control the GPU at the lowest level")
        print("• Optimization can significantly reduce instruction count")

    def run(self):
        """Run the interactive lesson."""
        try:
            for _, func in self.lessons:
                func()
                self.wait_for_enter()

            print(f"\n{Colors.BOLD}{Colors.GREEN}Lesson Complete!{Colors.ENDC}")

        except KeyboardInterrupt:
            print(f"\n\n{Colors.YELLOW}Lesson interrupted. Goodbye!{Colors.ENDC}")
            sys.exit(0)


class SASSExplanations:
    """Database of SASS instruction explanations - shared across all lessons."""

    INSTRUCTIONS = {
        "MOV": {
            "name": "Move/Copy",
            "desc": "Copies data from source to destination",
            "detail": "MOV is the basic data movement instruction.",
        },
        "S2R": {
            "name": "Special to Register",
            "desc": "Reads special register value",
            "detail": "S2R reads hardware special registers like thread ID, block ID.",
            "special_regs": {
                "SR_CTAID.X": "Block ID in X dimension (blockIdx.x)",
                "SR_TID.X": "Thread ID in block (threadIdx.x)",
                "SR_LANEID": "Lane ID within warp (0-31)"
            }
        },
        "IMAD": {
            "name": "Integer Multiply-Add",
            "desc": "Performs dst = (src1 * src2) + src3",
            "detail": "IMAD is commonly used for address calculations.",
        },
        "ISETP": {
            "name": "Integer Set Predicate",
            "desc": "Sets predicate register based on comparison",
            "detail": "ISETP performs integer comparison and sets a predicate register.",
        },
        "BRA": {
            "name": "Branch",
            "desc": "Jumps to target label",
            "detail": "BRA changes program flow. Can be conditional with @P predicate.",
        },
        "LDG": {
            "name": "Load Global",
            "desc": "Loads data from global memory",
            "detail": "LDG reads from global memory.",
            "modifiers": [".E (extended)", ".CONSTANT (read-only)"]
        },
        "STG": {
            "name": "Store Global",
            "desc": "Stores data to global memory",
            "detail": "STG writes to global memory.",
        },
        "LDC": {
            "name": "Load Constant",
            "desc": "Loads from constant memory",
            "detail": "LDC reads from constant memory bank.",
        },
        "IADD3": {
            "name": "3-Input Integer Add",
            "desc": "Adds three integer values",
            "detail": "IADD3 can add three values in one instruction.",
        },
        "FADD": {
            "name": "Float Add",
            "desc": "Adds two floating-point values",
            "detail": "FADD performs single-precision floating-point addition.",
        },
        "EXIT": {
            "name": "Exit",
            "desc": "Terminates thread execution",
            "detail": "EXIT ends the current thread.",
        },
        # Add more instructions as needed for other lessons
        "LDS": {
            "name": "Load Shared",
            "desc": "Loads from shared memory",
            "detail": "LDS reads from shared memory within a block.",
        },
        "STS": {
            "name": "Store Shared",
            "desc": "Stores to shared memory",
            "detail": "STS writes to shared memory within a block.",
        },
        "BAR": {
            "name": "Barrier",
            "desc": "Synchronization barrier",
            "detail": "BAR.SYNC synchronizes threads within a block.",
        },
        "SHFL": {
            "name": "Shuffle",
            "desc": "Warp shuffle operation",
            "detail": "SHFL exchanges data between threads in a warp.",
        },
        "ATOM": {
            "name": "Atomic",
            "desc": "Atomic memory operation",
            "detail": "ATOM performs atomic read-modify-write operations.",
        },
    }

    @classmethod
    def get_explanation(cls, mnemonic: str) -> Dict:
        """Get explanation for an instruction."""
        base = mnemonic.split('.')[0]
        return cls.INSTRUCTIONS.get(base, {
            "name": mnemonic,
            "desc": "Instruction specific to this lesson",
            "detail": "Refer to lesson documentation for details."
        })