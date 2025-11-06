#!/usr/bin/env python3
"""
Lesson 01: Vector Addition - Interactive Tutorial
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sass_lessons.base import BaseTutorial, Colors


class Lesson01Tutorial(BaseTutorial):
    """Interactive tutorial for Lesson 01: Vector Addition."""

    def get_title(self) -> str:
        return "VECTOR ADDITION"

    def get_description(self) -> str:
        return """This lesson teaches fundamental GPU operations through a simple vector addition kernel.
You'll understand how the GPU loads data, performs arithmetic, and stores results at the assembly level."""

    def _create_lessons(self):
        """Create the lesson sequence for vector addition."""
        return [
            ("Welcome", self.intro),
            ("The CUDA Kernel", self.show_cuda_kernel),
            ("GPU Architecture Basics", self.gpu_basics),
            ("Thread Indexing", self.thread_indexing),
            ("Loading Parameters", self.load_parameters),
            ("Calculate Thread Index", self.calculate_thread_index),
            ("Bounds Check", self.bounds_check),
            ("Memory Operations", self.memory_operations),
            ("Computation", self.computation),
            ("Optimized Version", self.optimized_version),
            ("Key Patterns", self.key_patterns),
            ("Performance Insights", self.performance),
            ("Summary", self.summary),
        ]

    def show_cuda_kernel(self):
        """Show the CUDA kernel code."""
        self.clear_screen()
        self.print_header("THE CUDA KERNEL")

        code = """extern "C" __global__
void vec_add(const float* __restrict__ a,
             const float* __restrict__ b,
             float* __restrict__ c,
             int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}"""
        self.print_code_block(code, "CUDA C++", width=70)  # 调整为70字符宽

        self.print_explanation("""This kernel does three things:
1. Calculates which array element this thread handles
2. Checks if the index is within bounds
3. Adds two floats and stores the result""")

    def gpu_basics(self):
        """Explain GPU architecture basics."""
        self.clear_screen()
        self.print_header("GPU ARCHITECTURE BASICS")

        print(f"{Colors.BOLD}Registers:{Colors.ENDC}")
        print(f"  {Colors.CYAN}R0-R254{Colors.ENDC}: General purpose (32-bit)")
        print(f"  {Colors.CYAN}RZ{Colors.ENDC}: Zero register (always 0)")
        print(f"  {Colors.CYAN}P0-P6{Colors.ENDC}: Predicate registers (1-bit)")
        print(f"  {Colors.CYAN}PT{Colors.ENDC}: Predicate true (always 1)")

        print(f"\n{Colors.BOLD}Memory Spaces:{Colors.ENDC}")
        print(f"  {Colors.CYAN}Global Memory{Colors.ENDC}: Main GPU memory (large, slower)")
        print(f"  {Colors.CYAN}Constant Memory{Colors.ENDC}: Cached, read-only (kernel params)")
        print(f"  {Colors.CYAN}Special Registers{Colors.ENDC}: Hardware values (thread/block IDs)")

        print(f"\n{Colors.BOLD}Thread Organization:{Colors.ENDC}")
        print(f"  • Each thread has {Colors.CYAN}threadIdx.x{Colors.ENDC} within its block")
        print(f"  • Each block has {Colors.CYAN}blockIdx.x{Colors.ENDC} in the grid")
        print(f"  • Global ID = {Colors.YELLOW}blockIdx.x * blockDim.x + threadIdx.x{Colors.ENDC}")

    def thread_indexing(self):
        """Explain thread indexing."""
        self.clear_screen()
        self.print_header("THREAD INDEXING")

        print("Every GPU thread needs to know which data element to process.\n")

        print(f"{Colors.BOLD}The Magic Formula:{Colors.ENDC}")
        self.print_code_block("int i = blockIdx.x * blockDim.x + threadIdx.x;", "CUDA")

        print(f"\n{Colors.BOLD}Visual Example:{Colors.ENDC}")
        print("  Grid with 3 blocks, 4 threads per block:\n")
        print(f"  {Colors.CYAN}Block 0:{Colors.ENDC} [T0] [T1] [T2] [T3] → Global IDs: 0,1,2,3")
        print(f"  {Colors.CYAN}Block 1:{Colors.ENDC} [T0] [T1] [T2] [T3] → Global IDs: 4,5,6,7")
        print(f"  {Colors.CYAN}Block 2:{Colors.ENDC} [T0] [T1] [T2] [T3] → Global IDs: 8,9,10,11")

        self.print_explanation("\nEach thread computes a unique global index\nThis ensures no two threads process the same element")

    def load_parameters(self):
        """Explain loading kernel parameters."""
        self.clear_screen()
        self.print_header("SASS: LOADING KERNEL PARAMETERS")

        print("First, the GPU loads our kernel parameters from constant memory:\n")

        print(f"{Colors.BOLD}Unoptimized SASS (-O0):{Colors.ENDC}\n")

        self.print_instruction("MOV R1, c[0x0][0x28]", "Load stack pointer")
        self.print_explanation("MOV copies data from constant memory to register R1")
        self.wait_for_enter("Press [ENTER] for next instruction...")

        print()
        self.print_instruction("LDC.64 R2, c[0x0][R2+0x160]", "Load pointer 'a' (64-bit)")
        self.print_explanation("LDC = Load from Constant memory\n.64 = 64-bit load (pointers are 64-bit)\nc[0x0][0x160] = Constant memory bank 0, offset 0x160")

    def calculate_thread_index(self):
        """Explain thread index calculation."""
        self.clear_screen()
        self.print_header("SASS: CALCULATE GLOBAL THREAD INDEX")

        print("Now we calculate which array element this thread handles:\n")

        print(f"{Colors.BOLD}The SASS Instructions:{Colors.ENDC}\n")

        self.print_instruction("S2R R2, SR_CTAID.X", "Read blockIdx.x")
        self.print_explanation("S2R = Special Register to Register\nSR_CTAID.X contains the block ID")
        self.wait_for_enter()

        print()
        self.print_instruction("S2R R4, SR_TID.X", "Read threadIdx.x")
        self.print_explanation("SR_TID.X contains the thread ID within the block")
        self.wait_for_enter()

        print()
        self.print_instruction("IMAD R3, R2, c[0x0][0x0], R4", "Global index")
        self.print_explanation("IMAD = Integer Multiply-Add\nR3 = R2 * blockDim + R4\nThis gives us: blockIdx.x * blockDim.x + threadIdx.x")

        print(f"\n{Colors.YELLOW}This is THE pattern you'll see in every GPU kernel!{Colors.ENDC}")

    def bounds_check(self):
        """Explain bounds checking."""
        self.clear_screen()
        self.print_header("SASS: BOUNDS CHECK")

        print("GPUs use predication to handle conditional execution:\n")

        print(f"{Colors.BOLD}The Bounds Check:{Colors.ENDC}\n")

        self.print_instruction("ISETP.GE.AND P0, PT, R3, R0, PT", "Compare index >= n")
        self.print_explanation("ISETP = Integer Set Predicate\n.GE = Greater or Equal\nSets P0 = true if R3 >= R0 (index >= n)")
        self.wait_for_enter()

        print()
        self.print_instruction("@P0 BRA `(.L_x_0)", "Branch if out of bounds")
        self.print_explanation("@P0 means 'execute if P0 is true'\nBRA = Branch to label\nSkips computation if thread is out of bounds")

        print(f"\n{Colors.YELLOW}Key Insight:{Colors.ENDC}")
        print("GPUs prefer predication over branching when possible.")
        print("This avoids thread divergence within a warp.")

    def memory_operations(self):
        """Explain memory operations."""
        self.clear_screen()
        self.print_header("SASS: MEMORY OPERATIONS")

        print("Loading data from global memory:\n")

        print(f"{Colors.BOLD}Calculate Addresses:{Colors.ENDC}\n")
        self.print_instruction("IMAD.WIDE R4, R4, 0x4, RZ", "Byte offset")
        self.print_explanation("Multiply index by 4 (sizeof(float))\nRZ is the zero register")
        self.wait_for_enter()

        print()
        self.print_instruction("IADD3 R10, P0, R10, R3, RZ", "Address for a[i]")
        self.print_explanation("IADD3 = 3-input integer add\nAdds base pointer + offset")

        print(f"\n{Colors.BOLD}Load Values:{Colors.ENDC}\n")
        self.print_instruction("LDG.E.CONSTANT R5, [R6.64]", "Load b[i]")
        self.print_explanation("LDG = Load from Global memory\n.E = Extended addressing (64-bit)\n.CONSTANT = Hint for read-only data")
        self.wait_for_enter()

        print()
        self.print_instruction("LDG.E.CONSTANT R6, [R6.64]", "Load a[i]")
        self.print_explanation("Both loads can be in flight simultaneously")

    def computation(self):
        """Explain the computation."""
        self.clear_screen()
        self.print_header("SASS: COMPUTATION & STORE")

        print("The actual vector addition:\n")

        print(f"{Colors.BOLD}Add the Values:{Colors.ENDC}\n")
        self.print_instruction("FADD R5, R6, R5", "R5 = a[i] + b[i]")
        self.print_explanation("FADD = Floating-point Add\nSimple addition of two floats")

        print(f"\n{Colors.BOLD}Store the Result:{Colors.ENDC}\n")
        self.print_instruction("STG.E [R2.64], R5", "Store to c[i]")
        self.print_explanation("STG = Store to Global memory\nWrites result back to memory")

        print(f"\n{Colors.BOLD}Exit:{Colors.ENDC}\n")
        self.print_instruction("EXIT", "Thread terminates")
        self.print_explanation("Thread has completed its work")

    def optimized_version(self):
        """Show optimized version."""
        self.clear_screen()
        self.print_header("OPTIMIZED SASS (-O3)")

        print("The compiler dramatically optimizes our kernel:\n")

        print(f"{Colors.BOLD}Comparison:{Colors.ENDC}")
        print(f"  {Colors.RED}Unoptimized (-O0):{Colors.ENDC} ~30+ instructions")
        print(f"  {Colors.GREEN}Optimized (-O3):{Colors.ENDC} ~11 instructions\n")

        print(f"{Colors.BOLD}Key Optimizations:{Colors.ENDC}\n")
        print("1. Early Exit - Bounds check immediately")
        print("2. Register Reuse - '.reuse' suffix")
        print("3. Parallel Loads - Back-to-back memory access")
        print("4. Direct Constants - No intermediate MOVs")
        print("5. Instruction Scheduling - Maximize parallelism")

    def key_patterns(self):
        """Show key SASS patterns."""
        self.clear_screen()
        self.print_header("KEY SASS PATTERNS")

        print("These patterns appear in almost every GPU kernel:\n")

        print(f"{Colors.BOLD}Pattern 1: Thread Index Calculation{Colors.ENDC}")
        self.print_code_block("""S2R Rx, SR_CTAID.X
S2R Ry, SR_TID.X
IMAD Rz, Rx, blockDim, Ry""", "SASS")

        print(f"{Colors.BOLD}Pattern 2: Predicated Execution{Colors.ENDC}")
        self.print_code_block("""ISETP.GE.AND P0, PT, index, limit, PT
@P0 EXIT  # or @!P0 for inverse""", "SASS")

        print(f"{Colors.BOLD}Pattern 3: Global Memory Access{Colors.ENDC}")
        self.print_code_block("""LDG.E register, [address.64]   # Load
STG.E [address.64], register   # Store""", "SASS")

    def performance(self):
        """Explain performance considerations."""
        self.clear_screen()
        self.print_header("PERFORMANCE INSIGHTS")

        print(f"{Colors.BOLD}Memory Coalescing:{Colors.ENDC}")
        print("When consecutive threads access consecutive memory,")
        print("the hardware combines them into fewer transactions.\n")

        print(f"{Colors.BOLD}Register Pressure:{Colors.ENDC}")
        print(f"  {Colors.RED}Unoptimized:{Colors.ENDC} 15 registers per thread")
        print(f"  {Colors.GREEN}Optimized:{Colors.ENDC} 12 registers per thread")
        print("  → More threads can run simultaneously\n")

        print(f"{Colors.BOLD}Instruction Latency Hiding:{Colors.ENDC}")
        print("GPU switches between warps while waiting for memory")
        print("This hides the ~300 cycle memory latency")

    def show_key_takeaways(self):
        """Show key takeaways specific to lesson 1."""
        print(f"{Colors.BOLD}What You've Learned:{Colors.ENDC}\n")

        print("✓ How kernel parameters pass through constant memory")
        print("✓ How threads calculate their global index")
        print("✓ How predication enables conditional execution")
        print("✓ How global memory loads and stores work")
        print("✓ How compiler optimization reduces instructions\n")

        print(f"{Colors.BOLD}Key Takeaways:{Colors.ENDC}\n")
        print("• SASS is the actual GPU machine code")
        print("• Simple operations become many instructions")
        print("• Optimization can reduce instructions by 3x")
        print("• Memory access patterns matter for performance")
        print("• These patterns appear in every GPU kernel")


def main():
    """Run the tutorial standalone."""
    tutorial = Lesson01Tutorial()
    tutorial.run()


if __name__ == "__main__":
    main()