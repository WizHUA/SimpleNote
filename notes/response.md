
=== Scatter

`Scatter`的函数原型如下：

```c
MPI_Scatter(
    void* send_data,
    int send_count,
    MPI_Datatype send_type,
    void* recv_data,
    int recv_count,
    MPI_Datatype recv_type,
    int root,
    MPI_Comm communicator)
```\ #v(-16pt)

其中：`send_data`参数是只在根进程上有效的待分发数据数组。`recv_data`是所有进程接收数据的缓冲区。`send_count`和`recv_count`分别指定发送和接收的数据元素数量。`root`指定分发数据的根进程，`communicator`指定参与通信的进程组。

Open MPI实现了多种Scatter算法：

==== 二项式树算法（Binomial Tree Algorithm）

*函数*：`ompi_coll_base_scatter_intra_binomial()`

源码文件路径：#link("ompi/mca/coll/base/coll_base_scatter.c")

其主要原理是：使用二项式树结构递归分发数据，根进程逐层向下传递数据块，每个内部节点收到数据后保留自己的部分，并将剩余数据继续向子节点分发。

#let code = ```c
int ompi_coll_base_scatter_intra_binomial(
    const void *sbuf, size_t scount, struct ompi_datatype_t *sdtype,
    void *rbuf, size_t rcount, struct ompi_datatype_t *rdtype,
    int root, struct ompi_communicator_t *comm,
    mca_coll_base_module_t *module)
{
    // 创建二项式树
    COLL_BASE_UPDATE_IN_ORDER_BMTREE(comm, base_module, root);
    ompi_coll_tree_t *bmtree = data->cached_in_order_bmtree;
    
    vrank = (rank - root + size) % size;
    
    if (vrank % 2) {  // 叶节点
        // 从父进程接收数据
        err = MCA_PML_CALL(recv(rbuf, rcount, rdtype, bmtree->tree_prev,
                                MCA_COLL_BASE_TAG_SCATTER, comm, &status));
        return MPI_SUCCESS;
    }
    
    // 根进程和内部节点处理数据
    if (rank == root) {
        curr_count = scount * size;
        // 数据重排序以适应分发顺序
        if (0 != root) {
            // 对非0根进程进行数据旋转
            opal_convertor_pack(&convertor, iov, &iov_size, &max_data);
        }
    } else {
        // 非根内部节点：从父进程接收数据
        err = MCA_PML_CALL(recv(ptmp, packed_size, MPI_PACKED, bmtree->tree_prev,
                                MCA_COLL_BASE_TAG_SCATTER, comm, &status));
        curr_count = status._ucount;
    }
    
    // 本地复制自己需要的数据
    if (rbuf != MPI_IN_PLACE) {
        err = ompi_datatype_sndrcv(ptmp, scount, sdtype,
                                   rbuf, rcount, rdtype);
    }
    
    // 向子节点发送数据
    for (int i = bmtree->tree_nextsize - 1; i >= 0; i--) {
        int vchild = (bmtree->tree_next[i] - root + size) % size;
        int send_count = vchild - vrank;
        if (send_count > size - vchild)
            send_count = size - vchild;
        send_count *= scount;
        
        err = MCA_PML_CALL(send(ptmp + (curr_count - send_count) * sextent,
                                send_count, sdtype, bmtree->tree_next[i],
                                MCA_COLL_BASE_TAG_SCATTER,
                                MCA_PML_BASE_SEND_STANDARD, comm));
        curr_count -= send_count;
    }
    
    return MPI_SUCCESS;
}
```

#figure(
  code,
  caption: [二项式树Scatter算法核心代码]
)\ #v(-16pt)

算法复杂度分析：二项式树散射算法的时间复杂度为$O(αlog(p) + βm(p-1)/p)$，其中$m = "scount" × "comm_size"$为总数据量。延迟复杂度为$O(log p)$，相比线性算法的$O(p)$有显著改善；带宽复杂度为$O(m(p-1)/p)$，当进程数较大时接近$O(m)$的最优效率。算法内存需求因角色而异：根进程需要$"scount" × "comm_size" × "sdtype_size"$内存，非根非叶进程需要$"rcount" × "comm_size" × "rdtype_size"$内存。

适用场景包括大规模通信子($p>8$)、大消息分发、延迟敏感应用，以及需要避免根进程成为瓶颈的场景。该算法通过树形结构有效分担根进程负载，在通信轮数和带宽利用率之间达到良好平衡，特别适合高性能计算中的大规模数据分发操作。

==== 线性算法（Linear Algorithm）

*函数*：`ompi_coll_base_scatter_intra_basic_linear()`

源码文件路径：#link("ompi/mca/coll/base/coll_base_scatter.c")

其主要原理是：根进程顺序向每个进程发送对应的数据块，所有非根进程直接从根进程接收数据。

#let code = ```c
int ompi_coll_base_scatter_intra_basic_linear(
    const void *sbuf, size_t scount, struct ompi_datatype_t *sdtype,
    void *rbuf, size_t rcount, struct ompi_datatype_t *rdtype,
    int root, struct ompi_communicator_t *comm,
    mca_coll_base_module_t *module)
{
    int i, rank, size, err;
    ptrdiff_t incr;
    char *ptmp;
    
    rank = ompi_comm_rank(comm);
    size = ompi_comm_size(comm);
    
    // 非根进程：接收数据
    if (rank != root) {
        err = MCA_PML_CALL(recv(rbuf, rcount, rdtype, root,
                                MCA_COLL_BASE_TAG_SCATTER,
                                comm, MPI_STATUS_IGNORE));
        return err;
    }
    
    // 根进程：循环发送数据
    err = ompi_datatype_type_extent(sdtype, &incr);
    incr *= scount;
    
    for (i = 0, ptmp = (char *)sbuf; i < size; ++i, ptmp += incr) {
        if (i == rank) {
            // 简单优化：根进程本地复制
            if (MPI_IN_PLACE != rbuf) {
                err = ompi_datatype_sndrcv(ptmp, scount, sdtype, 
                                           rbuf, rcount, rdtype);
            }
        } else {
            // 向其他进程发送数据
            err = MCA_PML_CALL(send(ptmp, scount, sdtype, i,
                                    MCA_COLL_BASE_TAG_SCATTER,
                                    MCA_PML_BASE_SEND_STANDARD, comm));
        }
        if (MPI_SUCCESS != err) return err;
    }
    
    return MPI_SUCCESS;
}
```

#figure(
  code,
  caption: [线性Scatter算法核心代码]
)\ #v(-16pt)

#let add = [
  根据源码注释，线性算法被从BASIC组件复制到BASE组件中，主要原因是：
  1. 算法简单且不进行消息分段
  2. 对于小规模节点或小数据量，性能与复杂的树形分段算法相当
  3. 可被决策函数选择作为特定场景的最优选择
  4. V1版本的模块选择机制要求代码复制，V2版本将采用不同方式处理
]

算法复杂度分析：线性散射算法的时间复杂度为$O((p-1)α + (p-1)βm')$，其中$m' = "scount"$为单个数据块大小。延迟复杂度为$O(p)$，根进程需要进行$p-1$次串行发送操作；带宽复杂度为$O(pm')$，总传输量为所有数据块之和。该算法实现最为简单#footnote(add)，无需构建树形拓扑，空间复杂度为$O(1)$。

适用场景包括小规模通信子($p≤4$)、小消息分发、网络连接性差的环境，以及作为复杂算法的回退选择。该算法的主要优势是实现简单、无拓扑构建开销，但在大规模场景下根进程会成为严重瓶颈，扩展性较差。

==== 非阻塞线性算法（Linear Non-blocking Algorithm）

*函数*：`ompi_coll_base_scatter_intra_linear_nb()`

源码文件路径：#link("ompi/mca/coll/base/coll_base_scatter.c")

其主要原理是：使用非阻塞发送(`isend`)分发数据，并通过周期性的阻塞发送来刷新本地资源，确保通信进展的同时避免资源耗尽。

#let code = ```c
int ompi_coll_base_scatter_intra_linear_nb(
    const void *sbuf, size_t scount, struct ompi_datatype_t *sdtype,
    void *rbuf, size_t rcount, struct ompi_datatype_t *rdtype,
    int root, struct ompi_communicator_t *comm,
    mca_coll_base_module_t *module, int max_reqs)
{
    int i, rank, size, err, nreqs;
    ompi_request_t **reqs = NULL, **preq;
    
    rank = ompi_comm_rank(comm);
    size = ompi_comm_size(comm);
    
    // 非根进程：接收数据
    if (rank != root) {
        err = MCA_PML_CALL(recv(rbuf, rcount, rdtype, root,
                                MCA_COLL_BASE_TAG_SCATTER,
                                comm, MPI_STATUS_IGNORE));
        return MPI_SUCCESS;
    }
    
    // 计算请求数量和分配请求数组
    if (max_reqs <= 1) {
        nreqs = size - 1;  // 除自己外的所有进程
    } else {
        // 周期性使用阻塞发送，减少请求数量
        nreqs = size - (size / max_reqs);
    }
    
    reqs = ompi_coll_base_comm_get_reqs(module->base_data, nreqs);
    
    // 根进程：循环发送数据
    for (i = 0, ptmp = (char *)sbuf, preq = reqs; i < size; ++i, ptmp += incr) {
        if (i == rank) {
            // 本地复制
            if (MPI_IN_PLACE != rbuf) {
                err = ompi_datatype_sndrcv(ptmp, scount, sdtype, 
                                           rbuf, rcount, rdtype);
            }
        } else {
            if (!max_reqs || (i % max_reqs)) {
                // 使用非阻塞发送
                err = MCA_PML_CALL(isend(ptmp, scount, sdtype, i,
                                         MCA_COLL_BASE_TAG_SCATTER,
                                         MCA_PML_BASE_SEND_STANDARD,
                                         comm, preq++));
            } else {
                // 周期性使用阻塞发送作为资源刷新
                err = MCA_PML_CALL(send(ptmp, scount, sdtype, i,
                                        MCA_COLL_BASE_TAG_SCATTER,
                                        MCA_PML_BASE_SEND_STANDARD, comm));
            }
        }
        if (MPI_SUCCESS != err) goto err_hndl;
    }
    
    // 等待所有非阻塞发送完成
    err = ompi_request_wait_all(preq - reqs, reqs, MPI_STATUSES_IGNORE);
    
    return MPI_SUCCESS;
}
```

#figure(
  code,
  caption: [非阻塞线性Scatter算法核心代码]
)\ #v(-16pt)

#let add = [
  `max_reqs`参数的作用机制：
  1. 当`max_reqs ≤ 1`时，所有发送都使用非阻塞方式
  2. 当`max_reqs > 1`时，每`max_reqs`个发送操作中插入一次阻塞发送
  3. 阻塞发送起到"本地资源刷新"的作用，确保通信进展并避免缓冲区溢出
  4. 这种混合策略在性能和资源利用之间达到平衡
]

算法复杂度分析：非阻塞线性散射算法的时间复杂度与标准线性算法相同，为$O((p-1)α + (p-1)βm')$，但通过非阻塞通信获得更好的重叠效果。延迟复杂度理论上仍为$O(p)$，但实际延迟因通信重叠而降低；带宽复杂度为$O(pm')$。该算法通过`max_reqs`参数#footnote(add)控制资源使用，在内存需求和性能之间提供可调节的权衡。

适用场景包括中等规模通信子、需要通信-计算重叠的应用、内存资源有限但希望改善性能的环境，以及网络具有良好并发处理能力的场景。该算法在保持线性算法简单性的同时，通过非阻塞技术提升了性能，是资源受限环境下的良好选择。

==== 其它Scatter算法

除了上述实现的算法外，Open MPI的Scatter操作在实际应用中还可能采用以下策略：

#list(
[基于网络拓扑的优化算法\
根据具体的网络拓扑结构（如胖树、环面等）优化数据分发路径，充分利用网络带宽和减少拥塞。],

[混合算法策略\
根据消息大小和进程数量动态选择算法，小消息使用线性算法避免拓扑开销，大消息使用树形算法提高效率。],

[分段传输优化\
对于超大消息，采用分段传输策略，结合流水线技术实现更好的内存利用和通信重叠。],

[容错增强算法\
在不可靠网络环境中，添加错误检测和恢复机制，确保数据分发的可靠性。] )\ #v(-16pt)

这些算法的设计体现了Open MPI在不同应用场景下的适应性。通过算法选择机制，系统能够根据运行时参数（通信子大小、消息长度、网络特性等）自动选择最优的实现，为用户提供透明而高效的散射操作。
