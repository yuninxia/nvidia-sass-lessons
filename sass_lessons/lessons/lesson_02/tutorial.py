#!/usr/bin/env python3
"""
Lesson 02: Memory Access Patterns - Interactive Tutorial
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sass_lessons.base import BaseTutorial, Colors


class Lesson02Tutorial(BaseTutorial):
    """Interactive tutorial for Lesson 02: Memory Access Patterns."""

    def get_title(self) -> str:
        return "MEMORY ACCESS PATTERNS"

    def get_description(self) -> str:
        return """This lesson explores how different memory access patterns affect SASS code generation
and ultimately impact performance. You'll learn about coalesced vs. strided access."""

    def _create_lessons(self):
        """Create the lesson sequence for memory patterns."""
        return [
            ("Welcome", self.intro),
            ("Memory Coalescing Basics", self.coalescing_basics),
            ("The CUDA Kernels", self.show_cuda_kernels),
            ("SASS Analysis", self.sass_analysis),
            ("Performance Impact", self.performance_impact),
            ("Summary", self.summary),
        ]

    def coalescing_basics(self):
        """Explain memory coalescing."""
        self.clear_screen()
        self.print_header("MEMORY COALESCING BASICS")

        print(f"{Colors.BOLD}What is Memory Coalescing?{Colors.ENDC}\n")
        print("When consecutive threads access consecutive memory locations,")
        print("the GPU hardware can combine these accesses into fewer transactions.\n")

        print(f"{Colors.BOLD}Coalesced Access Pattern:{Colors.ENDC}")
        print("  Thread 0 → Memory[0]")
        print("  Thread 1 → Memory[1]")
        print("  Thread 2 → Memory[2]")
        print("  Thread 3 → Memory[3]")
        print(f"  {Colors.GREEN}→ Single 128-byte transaction!{Colors.ENDC}\n")

        print(f"{Colors.BOLD}Strided Access Pattern:{Colors.ENDC}")
        print("  Thread 0 → Memory[0]")
        print("  Thread 1 → Memory[32]")
        print("  Thread 2 → Memory[64]")
        print("  Thread 3 → Memory[96]")
        print(f"  {Colors.RED}→ Multiple transactions!{Colors.ENDC}")

    def show_cuda_kernels(self):
        """Show the CUDA kernel variants."""
        self.clear_screen()
        self.print_header("THE CUDA KERNELS")

        print(f"{Colors.BOLD}Coalesced Access Kernel:{Colors.ENDC}")
        code = """__global__ void coalesced_copy(float* out, const float* in, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        out[tid] = in[tid];  // Consecutive threads, consecutive memory
    }
}"""
        self.print_code_block(code, "CUDA")

        print(f"{Colors.BOLD}Strided Access Kernel:{Colors.ENDC}")
        code = """__global__ void strided_copy(float* out, const float* in, int n, int stride) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        out[tid * stride] = in[tid * stride];  // Strided access
    }
}"""
        self.print_code_block(code, "CUDA")

    def sass_analysis(self):
        """Analyze SASS differences."""
        self.clear_screen()
        self.print_header("SASS ANALYSIS")

        print(f"{Colors.BOLD}Key SASS Differences:{Colors.ENDC}\n")

        print("1. Address Calculation:")
        print("   Coalesced: Simple offset addition")
        print("   Strided: Additional multiplication by stride\n")

        print("2. Memory Instructions:")
        print("   Both use LDG.E and STG.E")
        print("   But performance characteristics differ\n")

        print("3. Instruction Count:")
        print("   Strided version has extra IMAD for stride calculation")

        self.print_explanation("\nThe SASS looks similar, but runtime performance\ncan differ by 10x due to memory subsystem behavior!")

    def performance_impact(self):
        """Explain performance impact."""
        self.clear_screen()
        self.print_header("PERFORMANCE IMPACT")

        print(f"{Colors.BOLD}Memory Bandwidth Utilization:{Colors.ENDC}\n")

        print("Coalesced Access:")
        print("  • 32 threads request 128 bytes")
        print("  • Hardware fetches 128 bytes")
        print(f"  • Efficiency: {Colors.GREEN}100%{Colors.ENDC}\n")

        print("Strided Access (stride=32):")
        print("  • 32 threads request 128 bytes")
        print("  • Hardware fetches 32 * 128 = 4096 bytes")
        print(f"  • Efficiency: {Colors.RED}3.125%{Colors.ENDC}\n")

        print(f"{Colors.YELLOW}Key Insight:{Colors.ENDC}")
        print("SASS doesn't show the full performance picture.")
        print("Memory access patterns are critical for GPU performance!")

    def show_key_takeaways(self):
        """Show key takeaways specific to lesson 2."""
        print(f"{Colors.BOLD}What You've Learned:{Colors.ENDC}\n")

        print("✓ Memory coalescing combines multiple accesses")
        print("✓ Consecutive access patterns are optimal")
        print("✓ Strided access wastes memory bandwidth")
        print("✓ SASS shows instructions but not memory efficiency")
        print("✓ Access patterns can affect performance by 10x+\n")

        print(f"{Colors.BOLD}Best Practices:{Colors.ENDC}\n")
        print("• Structure data for coalesced access")
        print("• Use Structure of Arrays (SoA) not Array of Structures (AoS)")
        print("• Minimize stride in memory access patterns")


def main():
    """Run the tutorial standalone."""
    tutorial = Lesson02Tutorial()
    tutorial.run()


if __name__ == "__main__":
    main()