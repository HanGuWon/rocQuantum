# Future Roadmap for rocQuantum-1

This document outlines the strategic vision for the future development of the rocQuantum-1 simulator. The goals are organized into short-term, mid-term, and long-term milestones.

## Short-Term Goals (Next 3-6 Months)

*   ### Expand Gate Set
    *   **Multi-Controlled Gates:** Implement efficient HIP kernels for multi-controlled gates like the Toffoli (CCX) and Fredkin (CSWAP) gates.
    *   **Controlled Rotation Gates:** Add support for controlled rotation gates (CRX, CRY, CRZ), which are crucial for many quantum algorithms.

*   ### Performance Optimization
    *   **Gate Fusion:** Investigate and implement techniques for fusing sequential gates into a single, larger unitary. This will reduce kernel launch overhead and improve memory access patterns, leading to significant speedups for shallow circuits.
    *   **Memory Optimization:** Analyze and optimize GPU memory usage, particularly for intermediate data structures.

## Mid-Term Goals (6-12 Months)

*   ### Advanced Measurement Capabilities
    *   **Custom Measurement Operators:** Allow users to define and measure arbitrary Hamiltonians and operators beyond the simple Pauli basis.
    *   **Mid-Circuit Measurements:** Implement the ability to perform measurements in the middle of a circuit execution and conditionally apply subsequent gates based on the outcomes.

*   ### Noise Modeling
    *   **Basic Noise Channels:** Introduce support for common device noise models, such as depolarizing noise, bit-flip, and phase-flip channels. This will enable more realistic simulations of NISQ-era hardware.

## Long-Term Goals (1-2 Years and Beyond)

*   ### Multi-GPU Support
    *   **State Vector Parallelism:** Scale the simulator to utilize multiple GPUs on a single node. This will involve distributing the state vector across GPUs and managing inter-GPU communication (e.g., using `hipMemcpyPeerAsync`).
    *   **Cluster-Level Scaling:** Investigate the potential for scaling simulations across multiple nodes in a cluster using technologies like MPI.

*   ### Official PyPI Packaging
    *   **Automated Builds:** Create a robust CI/CD pipeline to build and test the C++/HIP code for various ROCm versions.
    *   **Easy Installation:** Package the simulator and Python bindings into an official package on PyPI. This will allow users to install it easily via `pip install rocquantum`, abstracting away the complexities of the build process.
