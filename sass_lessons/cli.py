#!/usr/bin/env python3
"""
Command-line interface for NVIDIA SASS Lessons
"""

import os
import sys
import argparse
from pathlib import Path


def find_sass_files(lesson_num: int = 1, arch: str = "sm_80") -> tuple[Path, Path]:
    """Find SASS files for a given lesson."""
    # Try to find project root
    current = Path.cwd()

    # Look for disasm directory
    disasm_dir = None
    for parent in [current, current.parent, current.parent.parent]:
        if (parent / "disasm").exists():
            disasm_dir = parent / "disasm"
            break

    if not disasm_dir:
        print("Error: Could not find disasm directory. Please run from project root.")
        sys.exit(1)

    # Construct file paths
    unoptimized = disasm_dir / f"lesson_{lesson_num:02d}.{arch}.O0.sass"
    optimized = disasm_dir / f"lesson_{lesson_num:02d}.{arch}.O3.sass"

    return unoptimized, optimized


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="NVIDIA SASS Lessons - Interactive GPU Assembly Learning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  sass-dissect tutorial              # Run interactive tutorial for lesson 1
  sass-dissect explore               # Explore SASS code for lesson 1
  sass-dissect explore --optimized   # Explore optimized SASS
  sass-dissect build                 # Build SASS files
  sass-dissect --lesson 2 tutorial   # Run tutorial for lesson 2
        """
    )

    parser.add_argument(
        "--lesson", "-l",
        type=int,
        default=1,
        help="Lesson number (default: 1)"
    )

    parser.add_argument(
        "--arch",
        default="sm_80",
        help="GPU architecture (default: sm_80)"
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Tutorial command
    tutorial_parser = subparsers.add_parser("tutorial", help="Run interactive tutorial")

    # Explorer command
    explore_parser = subparsers.add_parser("explore", help="Explore SASS assembly")
    explore_parser.add_argument(
        "--optimized", "-O3",
        action="store_true",
        help="Explore optimized version"
    )

    # Build command
    build_parser = subparsers.add_parser("build", help="Build SASS files")

    # List command
    list_parser = subparsers.add_parser("list", help="List available lessons")

    args = parser.parse_args()

    if not args.command:
        print("Welcome to NVIDIA SASS Lessons!")
        print("\nChoose an option:")
        print("  1. Interactive Tutorial")
        print("  2. SASS Explorer (Unoptimized)")
        print("  3. SASS Explorer (Optimized)")
        print("  4. Build SASS Files")
        print("  5. List Lessons")
        print("\nEnter choice [1-5]: ", end="")

        try:
            choice = input().strip()
            if choice == "1":
                args.command = "tutorial"
            elif choice == "2":
                args.command = "explore"
            elif choice == "3":
                args.command = "explore"
                args.optimized = True
            elif choice == "4":
                args.command = "build"
            elif choice == "5":
                args.command = "list"
            else:
                print("Invalid choice")
                sys.exit(1)
        except KeyboardInterrupt:
            print("\nGoodbye!")
            sys.exit(0)

    # Execute commands
    if args.command == "tutorial":
        # Import here to avoid circular imports
        from sass_lessons.tutorial import run_tutorial

        print(f"Starting interactive tutorial for Lesson {args.lesson:02d}...")
        run_tutorial(args.lesson)

    elif args.command == "explore":
        from sass_lessons.explorer import SASSExplorer

        unopt_file, opt_file = find_sass_files(args.lesson, args.arch)
        sass_file = opt_file if getattr(args, 'optimized', False) else unopt_file

        if not sass_file.exists():
            print(f"Error: SASS file not found: {sass_file}")
            print("Run 'sass-dissect build' first to generate SASS files.")
            sys.exit(1)

        print(f"Exploring: {sass_file.name}")
        explorer = SASSExplorer(str(sass_file))
        explorer.run()

    elif args.command == "build":
        # Try to find and run build script
        build_script = None
        for parent in [Path.cwd(), Path.cwd().parent]:
            script_path = parent / "scripts" / "build.sh"
            if script_path.exists():
                build_script = script_path
                break

        if not build_script:
            print("Error: Could not find build.sh script")
            sys.exit(1)

        print(f"Building SASS files for {args.arch}...")
        os.system(f"SASS_ARCH={args.arch} bash {build_script}")

    elif args.command == "list":
        # Find all lessons
        lessons_dir = Path(__file__).parent / "lessons"
        if not lessons_dir.exists():
            # Fallback to old location if needed
            project_root = Path.cwd()
            lessons_dir = project_root / "sass_lessons" / "lessons"

        lessons = sorted(lessons_dir.glob("lesson_*"))

        print("\nAvailable Lessons:")
        print("-" * 50)

        lesson_info = {
            1: "Vector Addition - Basic GPU operations",
            2: "Memory Access - Coalescing patterns",
            3: "Shared Memory - Tiling and barriers",
            4: "Predication - Conditional execution",
            5: "Control Flow - Loops and branches",
            6: "Warp Shuffles - Register data exchange",
            7: "Atomics - Global atomic operations",
            8: "Barriers - Block synchronization",
            9: "Special Registers - Hardware values"
        }

        for lesson_dir in lessons:
            lesson_num = int(lesson_dir.name.split("_")[1])
            desc = lesson_info.get(lesson_num, "")
            status = "✓" if (lesson_dir / "kernel.cu").exists() else "✗"
            print(f"  {status} Lesson {lesson_num:02d}: {desc}")

        print("\nRun with: sass-dissect --lesson N tutorial")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()