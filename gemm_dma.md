1. 按 **tileM**、**tileN** 双循环划分子块（每个核负责最多 **64×64** 的 **C_ldm**，对应 **64×kTileSize** 的 **A_panel** 和 **kTileSize×64** 的 **B_panel**）。

**now=0**，**next=1**（仅用于描述，实际实现里不再区分两个缓冲，因为没有 RMA 双缓冲）

1. **_rid=0**、**_cid=0** 的核直接 DMA 自己负责的 **A_panel**、**B_panel** 到本地 **A_panel**、**B_panel**。
2. 等待这次 DMA 完成。
3. （无 RMA 步骤；所有核都只依赖自己的 DMA 数据。）
4. **_rid=0**、**_cid=0** 的核准备下一段 **A_panel**、**B_panel**（实作中在每个 K 子块循环开始时完成）。
5. 按 **K** 循环（每次处理 **kTileSize**=32）：

* 5.0 **now=next**，**next=1-now**（仅用于说明）。
* 5.1 等待本轮 DMA 完成后，用 **A_panel**、**B_panel** 做 SIMD 计算，累加到 **C_ldm**。
* 5.2 **_rid=0**、**_cid=0** 的核启动下一段 **A_panel**、**B_panel** 的 DMA（若 K 还有剩余）。
* 5.3 **_rid=0**、**_cid=0** 的核等待当前段 DMA 完成，继续进入下一轮计算（这一过程在代码里是“DMA → 等待 → 计算 → 下一次 DMA”的顺序）。

1. 剩余步骤：

* 6.1 对于最后一段，DMA 完成后直接使用 **A_panel** / **B_panel** 计算（不再有后续 RMA / DMA）。
* 6.2 等待这段 DMA 完成后，用其数据完成最后一轮乘加。

1. 所有 K 段累加完后，将 **C_ldm** 通过 DMA 写回到全局 **C_global**。
