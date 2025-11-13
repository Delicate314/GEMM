0.按tileM和tileN循环（每个核处理64x32的A和32X64的B）

now=0,next=1

1.rid=0,cid=0的核DMA 到A_stage[next],B_stage[next]

2.等待DMA完成

3.rid=0按列RMA，cid=0按行RMA到A_compute[next],B_compute[next]

4.rid=0,cid=0的核DMA到A_stage[now]和B_stage[now]

5.按K循环（每次处理kTileSize）:
	5.0. now=next,next=1-next

​	5.1. 等待RMA完成，使用A_compute[now]和B_compute[now]开始计算

​	5.2. (rid=0，cid=0)等待DMA完成，rid=0按列RMA，cid=0按行RMA到A_compute[next],B_compute[next]

​	5.36. rid=0和cid=0的核DMA到A_stage[now],B_stage[now]

6.剩余步骤：

​	6.1. rid=0按列RMA，cid=0按行RMA到A_compute[now],B_compute[now]

​	6.2. 等待RMA完成，开始计算A_compute[now],B_compute[now]

7.把C写回