0. 按 **tileM**、**tileN** 双循环划分子块（每个核负责最多 **64×64** 的 **C_ldm**）。

1. 清零当前子块的 **C_ldm**（使用 **memset**）。

2. 主循环：沿 **K** 维度按 **kSuperSize**（**8 × kTileSize = 256**）分块处理：

   **阶段1：领导核 DMA 数据**
   * 2.1 **列领导核**（**_cid=0**）DMA **A** 数据到 **A_stage**（大小为 **curM × curKSuper**）。
   * 2.2 **行领导核**（**_rid=0**）DMA **B** 数据到 **B_stage**（大小为 **curKSuper × curN**）。
   * 2.3 等待 DMA 完成（**列领导核**等待 **dma_rply_A**，**行领导核**等待 **dma_rply_B**）。

   **阶段2：RMA 广播**
   * 2.4 **列领导核**按列广播 **A_stage** → 所有核的 **A_compute**（使用 **athread_rma_col_ibcast**）。
   * 2.5 **行领导核**按行广播 **B_stage** → 所有核的 **B_compute**（使用 **athread_rma_row_ibcast**）。
   * 2.6 所有核等待 RMA 完成：
     * 先进行列同步（**athread_ssync(COL_SCOPE, 0xff)**），等待 **A** 的 RMA 完成。
     * 再进行行同步（**athread_ssync(ROW_SCOPE, 0xff)**），等待 **B** 的 RMA 完成。
   * 2.7 全局同步所有核（**athread_ssync()**）。

   **阶段3：SIMD 计算**
   * 2.8 将 **kSuperSize** 的数据按 **kTileSize**（32）分块计算：
     * 对于每个 **kTileSize** 子块，使用 **A_compute** 和 **B_compute** 进行 **SIMD 微内核**计算：
       * 外层循环：**i** 从 0 到 **curM**，步长 4（处理 4 行）。
       * 内层循环：**j** 从 0 到 **curN**，步长 8（处理 8 列，使用 **floatv8**）。
       * 最内层循环：**k_inner** 从 0 到 **curK**，累加到 **C_ldm**。
       * 使用 **simd_load**、**simd_vmas**、**simd_store** 等 SIMD 指令。

3. 所有 **K** 段累加完后，将 **C_ldm** 通过 DMA 写回到全局 **C_global**。

**说明**：
* 本实现**无流水线**，采用简单的 **DMA → RMA → 计算** 顺序。
* 每个领导核每次 DMA **8 倍**的 **K** 维度数据（**kSuperSize = 256**），然后通过 RMA 广播给其他核。
* 所有核使用 **A_compute** 和 **B_compute** 进行 **SIMD 微内核**计算（4×8 的块大小，使用 **floatv8** 向量）。
* RMA 广播使用两个回答字（本地回答字和远程回答字），并分别进行列同步和行同步。
