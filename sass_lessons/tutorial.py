#!/usr/bin/env python3
"""
Tutorial loader - dynamically loads lesson-specific tutorials
"""

import sys
import importlib.util
from pathlib import Path
from typing import Optional

from sass_lessons.base import BaseTutorial


def load_lesson_tutorial(lesson_num: int) -> Optional[BaseTutorial]:
    """
    Dynamically load a lesson's tutorial module.

    Args:
        lesson_num: The lesson number to load

    Returns:
        An instance of the lesson's tutorial class, or None if not found
    """
    # Find lessons directory
    current = Path(__file__).parent

    # Look for lesson directory in sass_lessons/lessons/
    lesson_dir = current / "lessons" / f"lesson_{lesson_num:02d}"
    if not lesson_dir.exists():
        print(f"Error: Lesson {lesson_num:02d} directory not found.")
        return None

    # Look for tutorial.py in the lesson directory
    tutorial_path = lesson_dir / "tutorial.py"
    if not tutorial_path.exists():
        print(f"Note: Lesson {lesson_num:02d} does not have an interactive tutorial yet.")
        print(f"Available files in {lesson_dir.name}:")
        for file in lesson_dir.iterdir():
            print(f"  - {file.name}")
        return None

    # Dynamically load the module
    try:
        # Create a unique module name
        module_name = f"lesson_{lesson_num:02d}_tutorial"

        # Load the module
        spec = importlib.util.spec_from_file_location(module_name, tutorial_path)
        if spec is None or spec.loader is None:
            print(f"Error: Could not load tutorial from {tutorial_path}")
            return None

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

        # Look for the tutorial class
        # Convention: LessonXXTutorial class
        class_name = f"Lesson{lesson_num:02d}Tutorial"

        if hasattr(module, class_name):
            tutorial_class = getattr(module, class_name)
            return tutorial_class(lesson_num)
        else:
            # Try to find any class that inherits from BaseTutorial
            for name in dir(module):
                obj = getattr(module, name)
                if (isinstance(obj, type) and
                    issubclass(obj, BaseTutorial) and
                    obj != BaseTutorial):
                    return obj(lesson_num)

            print(f"Error: No tutorial class found in {tutorial_path}")
            print(f"Expected class name: {class_name}")
            return None

    except Exception as e:
        print(f"Error loading tutorial for lesson {lesson_num:02d}: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_tutorial(lesson_num: int):
    """
    Run the tutorial for a specific lesson.

    Args:
        lesson_num: The lesson number to run
    """
    tutorial = load_lesson_tutorial(lesson_num)
    if tutorial:
        tutorial.run()
    else:
        print(f"\nTo create a tutorial for lesson {lesson_num:02d}:")
        print(f"1. Create lesson_{lesson_num:02d}/tutorial.py")
        print(f"2. Define a Lesson{lesson_num:02d}Tutorial class that inherits from BaseTutorial")
        print(f"3. Implement the required methods (_create_lessons, get_title, get_description)")


# For backwards compatibility
class InteractiveLesson:
    """Wrapper for backwards compatibility."""

    def __init__(self, lesson_num: int = 1):
        self.lesson_num = lesson_num

    def run(self):
        run_tutorial(self.lesson_num)


def main():
    """Main entry point when run directly."""
    import argparse

    parser = argparse.ArgumentParser(description="Run interactive SASS tutorial")
    parser.add_argument("--lesson", type=int, default=1, help="Lesson number")

    args = parser.parse_args()
    run_tutorial(args.lesson)


if __name__ == "__main__":
    main()