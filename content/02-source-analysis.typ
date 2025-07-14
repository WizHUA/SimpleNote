#import "../template.typ": *
#import "@preview/cetz:0.3.0": canvas, draw

= Open MPI集合通信算法源码分析

基于#link("https://www.open-mpi.org/")[The Open MPI Project]。（#link("https://github.com/WizHUA/ompi")[仓库链接]）

== Open MPI架构概述

Open MPI作为高性能计算领域的主流MPI实现，采用了分层的模块化组件架构(Modular Component Architecture, MCA)设计。整个架构分为三个核心层次：
#list(
  [*OPAL*(Open Portability Access Layer) 提供操作系统和硬件抽象],
  [*OMPI* 实现MPI标准的核心功能],
  [*OSHMEM* 提供OpenSHMEM支持],
)\ #v(-16pt)

这种分层设计确保了代码的模块化和可移植性，并且为集合通信算法的实现和优化提供了灵活的框架支持。

集合通信的核心实现位于`ompi/mca/coll/`目录下，包含了`base`、`basic`、`tuned`、`han`、`xhc`等多个专门化组件。这种组件化设计通过`mca_coll_base_comm_select()`机制，能够根据消息大小、进程数量、网络拓扑等运行时参数动态选择最优算法。

在此次任务中，我们专注于研究消息大小和进程数量这两个核心因素对集合通信算法选择和性能的影响，暂不涉及网络拓扑、硬件特性等复杂环境因素。因此，我们在源码分析部分将重点关注以下三个核心组件：

#list(
  [`base`组件 - 提供基础算法实现和算法选择框架],
  [`basic`组件 - 包含简单可靠的参考算法实现],
  [`tuned`组件 - 集成多种优化算法和智能选择机制]
)\ #v(-16pt)

MCA架构的另一个关键特性是其参数化配置系统，通过MCA参数可以在运行时动态调整算法选择策略、消息分片大小、通信拓扑等关键参数，同时MPI_T接口提供了运行时性能监控和参数调优的能力。这种设计不仅为我们的参数配置分析提供了完整的参数空间，也为机器学习模型的训练数据收集和在线预测部署提供了技术基础。通过深入分析这些组件的实现机制和参数影响，我们可以系统地理解集合通信性能的影响因素，为后续的数据集构建、特征工程和预测模型设计奠定坚实的理论基础。

== 集合通信框架分析

集合通信框架的核心实现位于 #link("ompi/mca/coll/base/")[`ompi/mca/coll/base/`] 目录，采用动态组件选择机制为每个通信子配置最优的集合通信实现。

=== 框架核心文件结构

框架的关键文件包括：
#list(
  [`coll_base_functions.h` - 定义基础算法接口 \ 
  该部分定义所有集合通信操作的函数原型和参数宏（`typedef enum COLLTYPE`）；提供算法实现的标准化接口；并声明各种拓扑结构的缓存机制，提供通用的工具函数和数据结构，如二叉树（binary tree）、二项树（binomial tree）、k进制树（k-nomial tree）、链式拓扑（chained tree）、流水线拓扑（pipeline）等。],
  [`coll_base_comm_select.c` - 实现组件选择机制 \
  该部分为每个通信子动态选择最优的集合通信组件；处理组件优先级和兼容性检查；支持运行时组件偏好设置（通过`comm->super.s_info`等机制）。],
  [`coll_base_util.h` - 工具函数定义 \
  该部分支持配置文件解析和参数处理；提供调试和监控支持。]
)\ #v(-16pt)

=== 核心文件分析

具体而言，粗略分析这部分代码可以观察到：

1. 框架支持MPI标准定义的22种集合通信操作，通过`COLLTYPE`枚举类型进行分类管理：


#let code = ```c
typedef enum COLLTYPE {
    ALLGATHER = 0,       ALLGATHERV,          ALLREDUCE,           
    ALLTOALL,            ALLTOALLV,           ALLTOALLW,           
    BARRIER,             BCAST,               EXSCAN,              
    GATHER,              GATHERV,             REDUCE,              
    REDUCESCATTER,       REDUCESCATTERBLOCK,  SCAN,                
    SCATTER,             SCATTERV,            NEIGHBOR_ALLGATHER,  
    NEIGHBOR_ALLGATHERV, NEIGHBOR_ALLTOALL,   NEIGHBOR_ALLTOALLV,  
    NEIGHBOR_ALLTOALLW,  COLLCOUNT            
} COLLTYPE_T;
```

#figure(
  code,
  caption: [`coll_base_functions.h`集合通信操作类型枚举定义]
) <cpp-example>\ #v(-16pt)

2. 每种集合通信操作都提供三个层次的接口：
#list(
  [阻塞接口：如BCAST_ARGS，标准的同步集合通信],
  [非阻塞接口：如IBCAST_ARGS，支持异步执行和重叠计算],
  [持久化接口：如BCAST_INIT_ARGS，支持MPI-4的持久化集合通信]
)\ #v(-16pt)

3. #[为每个操作提供了丰富的算法变体实现，以Broadcast操作为例，框架提供了10种不同的算法实现：

#let code = ```c
/* Bcast */
int ompi_coll_base_bcast_intra_generic(BCAST_ARGS, uint32_t count_by_segment, ompi_coll_tree_t* tree);
int ompi_coll_base_bcast_intra_basic_linear(BCAST_ARGS);
int ompi_coll_base_bcast_intra_chain(BCAST_ARGS, uint32_t segsize, int32_t chains);
int ompi_coll_base_bcast_intra_pipeline(BCAST_ARGS, uint32_t segsize);
int ompi_coll_base_bcast_intra_binomial(BCAST_ARGS, uint32_t segsize);
int ompi_coll_base_bcast_intra_bintree(BCAST_ARGS, uint32_t segsize);
int ompi_coll_base_bcast_intra_split_bintree(BCAST_ARGS, uint32_t segsize);
int ompi_coll_base_bcast_intra_knomial(BCAST_ARGS, uint32_t segsize, int radix);
int ompi_coll_base_bcast_intra_scatter_allgather(BCAST_ARGS, uint32_t segsize);
int ompi_coll_base_bcast_intra_scatter_allgather_ring(BCAST_ARGS, uint32_t segsize);
```

#figure(
  code,
  caption: [`coll_base_functions.h`中的`bcast`的算法实现变体]
)\ #v(-16pt)

同样地，Allreduce操作提供了7种算法，Allgather提供了8种算法，覆盖了从延迟优化到带宽优化的各种应用场景。]

4. 多数算法函数支持参数化配置，如：

#list(
  [segsize参数：控制消息分段大小，影响内存使用和流水线效率],
  [radix参数：控制树形算法的分支数，平衡通信轮数和并发度],
  [max_requests参数：控制并发请求数量，影响内存和网络资源使用],
)

5. 以及其它相关拓展和配置接口（如拓扑感知的算法优化等），此处略。

\ #v(-16pt)

通过上面的分析，结合官方文档#link("https://docs.open-mpi.org/en/v5.0.x/mca.html")[The Modular Component Architecture (MCA) — Open MPI 5.0.x documentation]，我们可以较好的理解Open MPI集合通信框架，并通过配置参数使用特定的算法实现来优化通信操作的性能。具体的参数配置分析见于#link(<chap4>)[章节4. 集合通信参数配置分析]。

== 集合通信操作示例

为了更好地理解Open MPI集合通信框架的工作原理，以一个具体的`MPI_Reduce`操作为例，粗略分析从用户调用到底层算法执行的几个关键过程。

=== 用户代码调用

在用户的代码中调用MPI函数，假设8进程对一个`int`数据执行Reduce操作，有如下示例代码：

#let code = ```c
// size = 8
int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);  
    
    int local_data = rank + 1;
    int result = 0;
    
    // 执行Reduce操作，求和到进程0
    MPI_Reduce(&local_data, &result, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    
    if (rank == 0) {
        printf("Sum result: %d\n", result);
    }
    
    MPI_Finalize();
    return 0;
}
```

#figure(
  code,
  caption: [`MPI_Reduce`示例代码]
)

=== 初始调用暂记

*该节存在较多困惑尚未解决，仅作记录用。实际的操作示例描述从 #link(<tag1>)[初始化阶段：组件选择机制]开始*

当`MPI_Init()`调用时，首先被`monitoring_prof.c`中的监控层拦截：

#let code = ```c
int MPI_Init(int* argc, char*** argv)
{
    // 1. 调用实际的MPI实现
    result = PMPI_Init(argc, argv);
    
    // 2. 获取通信子基本信息
    PMPI_Comm_size(MPI_COMM_WORLD, &comm_world_size);
    PMPI_Comm_rank(MPI_COMM_WORLD, &comm_world_rank);
    
    // 3. 初始化MPI_T监控接口
    MPI_T_init_thread(MPI_THREAD_SINGLE, &provided);
    MPI_T_pvar_session_create(&session);
    
    // 4. 注册集合通信监控变量
    init_monitoring_result("coll_monitoring_messages_count", &coll_counts);
    init_monitoring_result("coll_monitoring_messages_size", &coll_sizes);
    start_monitoring_result(&coll_counts);
    start_monitoring_result(&coll_sizes);
}
```

#figure(
  code,
  caption: [`MPI_Init()`的过程]
)\ #v(-16pt)

补充：这里所指的“实际的`MPI`实现”，区别于此处`monitoring_prof.c`中的`MPI_Init()`：\

1. 监控库定义: `MPI_Init` (拦截版本)
2. 真实库定义: MPI_Init (真实实现) 
3. 通过 `#pragma weak: PMPI_Init -> MPI_Init` (实际实现)，实际实现在`ompi/mpi/c/init.c.in`中通过模板文件实现，在编译时生成实际的c代码。 #footnote("报告中对此类实现相关的问题只作了简单的追踪，未进行进一步的探究")
4. 通过 `LD_PRELOAD`: 用户调用先到监控版本 

\ #v(-16pt)

`init.c.in`如下：

#let code = ```c
PROTOTYPE INT init(INT_OUT argc, ARGV argv)
{
    int err;
    int provided;
    int required = MPI_THREAD_SINGLE;

    // 1. 检查环境变量设置的线程级别
    if (OMPI_SUCCESS > ompi_getenv_mpi_thread_level(&required)) {
        required = MPI_THREAD_MULTIPLE;
    }

    // 2. 调用后端初始化函数
    if (NULL != argc && NULL != argv) {
        err = ompi_mpi_init(*argc, *argv, required, &provided, false);
    } else {
        err = ompi_mpi_init(0, NULL, required, &provided, false);
    }

    // 3. 错误处理
    if (MPI_SUCCESS != err) {
        // return ompi_errhandler_invoke(NULL, NULL, OMPI_ERRHANDLER_TYPE_COMM,
                                      err < 0 ? ompi_errcode_get_mpi_code(err) : err, 
                                      FUNC_NAME);
    }

    // 4. 初始化性能计数器
    SPC_INIT();
    return MPI_SUCCESS;
}
```

#figure(
  code,
  caption: [模板生成的`MPI_Init`实现]
)\ #v(-16pt)

在`ompi_mpi_init.c`中的核心初始化过程如下：

#let code = ```c
int ompi_mpi_init(int argc, char **argv, int requested, int *provided, bool reinit_ok)
{
    // 1. 状态检查和线程级别设置
    ompi_mpi_thread_level(requested, provided);

    // 2. 初始化MPI实例
    ret = ompi_mpi_instance_init(*provided, &ompi_mpi_info_null.info.super, 
                                MPI_ERRORS_ARE_FATAL, &ompi_mpi_instance_default, 
                                argc, argv);

    // 3. 初始化通信子子系统
    if (OMPI_SUCCESS != (ret = ompi_comm_init_mpi3())) {
        error = "ompi_mpi_init: ompi_comm_init_mpi3 failed";
        goto error;
    }

    // 4. 为预定义通信子选择集合通信组件
    if (OMPI_SUCCESS != (ret = mca_coll_base_comm_select(MPI_COMM_WORLD))) {
        error = "mca_coll_base_comm_select(MPI_COMM_WORLD) failed";
        goto error;
    }

    if (OMPI_SUCCESS != (ret = mca_coll_base_comm_select(MPI_COMM_SELF))) {
        error = "mca_coll_base_comm_select(MPI_COMM_SELF) failed";
        goto error;
    }

    // 5. 标记初始化完成
    opal_atomic_swap_32(&ompi_mpi_state, OMPI_MPI_STATE_INIT_COMPLETED);
}
```

#figure(
  code,
  caption: [`ompi_mpi_init.c`中的核心初始化过程]
)\ #v(-16pt)

=== 组件选择的核心机制

在`mca_coll_base_comm_select()`中执行具体的组件选择：

#let code = ```c
int mca_coll_base_comm_select(ompi_communicator_t *comm)
{ 
    // 1. 初始化通信子的集合通信结构
    comm->c_coll = (mca_coll_base_comm_coll_t*)calloc(1, sizeof(mca_coll_base_comm_coll_t));

    // 2. 查询所有可用的集合通信组件（basic, tuned, han, xhc等）
    selectable = check_components(&ompi_coll_base_framework.framework_components, comm);

    // 3. 按优先级排序并启用最高优先级组件
    for (item = opal_list_remove_first(selectable); 
         NULL != item; 
         item = opal_list_remove_first(selectable)) {

        mca_coll_base_avail_coll_t *avail = (mca_coll_base_avail_coll_t *) item;
        ret = avail->ac_module->coll_module_enable(avail->ac_module, comm);

        if (OMPI_SUCCESS == ret) {
            // 4. 设置函数指针到具体实现
            if (NULL == comm->c_coll->coll_reduce) {
                comm->c_coll->coll_reduce = avail->ac_module->coll_reduce;
                comm->c_coll->coll_reduce_module = avail->ac_module;
            }
            if (NULL == comm->c_coll->coll_allgather) {
                comm->c_coll->coll_allgather = avail->ac_module->coll_allgather;
                comm->c_coll->coll_allgather_module = avail->ac_module;
            }
            // ... 为所有集合通信操作设置函数指针

            opal_list_append(comm->c_coll->module_list, &avail->super);
        }
    }

    // 5. 验证完整性 - 确保所有必需的集合通信操作都有实现
    #define CHECK_NULL(what, comm, func) \
        ((what) = # func, NULL == (comm)->c_coll->coll_ ## func)

    if (CHECK_NULL(which_func, comm, allgather) ||
        CHECK_NULL(which_func, comm, allreduce) ||
        CHECK_NULL(which_func, comm, reduce) ||
        // ... 检查其他操作
        ) {
        opal_show_help("help-mca-coll-base.txt",
                       "comm-select:no-function-available", true, which_func);
        return OMPI_ERR_NOT_FOUND;
    }

    return OMPI_SUCCESS;
}
```

#figure(
  code,
  caption: [集合通信组件选择过程]
)

=== 初始化阶段：组件选择机制 <tag1>

在`MPI_Init()`调用时，系统为`MPI_COMM_WORLD`选择合适的集合通信组件，这将影响后续`MPI_Reduce`的实现方式。

从`init.c.in`模板开始的调用链：

#show raw.where(block: true): it => sourcecode(numbering: none)[#it]

```c
用户调用: MPI_Init(&argc, &argv)
    ↓
模板生成: init.c.in → MPI_Init() 
    ↓
核心初始化: ompi_mpi_init(*argc, *argv, required, &provided, false)
    ↓
通信子初始化: ompi_comm_init_mpi3() [创建MPI_COMM_WORLD]
    ↓
组件选择: mca_coll_base_comm_select(MPI_COMM_WORLD)
```
#show raw.where(block: true): it => sourcecode()[#it]

\ #v(-16pt)

在组件选择过程中，系统为`MPI_COMM_WORLD`设置reduce函数指针：

#let code = ```c
// 在mca_coll_base_comm_select()中
if (OMPI_SUCCESS == ret) {
    // 关键：为reduce操作设置函数指针
    if (NULL == comm->c_coll->coll_reduce) {
        comm->c_coll->coll_reduce = selected_module->coll_reduce;
        comm->c_coll->coll_reduce_module = selected_module;
    }
    // ... 其他集合通信操作的设置
}
```

#figure(
  code,
  caption: [为`MPI_Reduce`设置函数指针]
)\ #v(-16pt)

假设系统选择了`basic`组件，则根据通信子的大小决定通信算法，譬如：

#let code = ```c
// 在basic组件模块启用时
if (ompi_comm_size(comm) <= mca_coll_basic_crossover) {
    // 小规模通信子：使用线性算法
    BASIC_INSTALL_COLL_API(comm, basic_module, reduce, 
                           ompi_coll_base_reduce_intra_basic_linear);
} else {
    // 大规模通信子：使用对数算法
    BASIC_INSTALL_COLL_API(comm, basic_module, reduce, 
                           mca_coll_basic_reduce_log_intra);
}
```

#figure(
  code,
  caption: [`basic`组件的通信算法选择机制]
)\ #v(-16pt)

对于8进程的情况，`mca_coll_basic_crossover（config = 4）< 8`，则会选择对数算法。



=== `MPI_Reduce`的调用过程

当用户调用

`MPI_Reduce(&local_data, &result, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD)`

*第1步：MPI接口层*
#let code = ```c
// 在ompi/mpi/c/reduce.c中
int MPI_Reduce(const void *sendbuf, void *recvbuf, int count,
               MPI_Datatype datatype, MPI_Op op, int root, MPI_Comm comm)
{
    // 参数验证：检查buffer、datatype、root等参数
    if (MPI_PARAM_CHECK) { /* ... */ }
    
    // 关键调用：通过函数指针调用已选定的reduce实现
    return comm->c_coll->coll_reduce(sendbuf, recvbuf, count, datatype,
                                     op, root, comm, 
                                     comm->c_coll->coll_reduce_module);
}
```

#figure(
  code,
  caption: [MPI接口层的分发]
)\ #v(-16pt)

*第2步：组件实现层*

#let note = [如果系统选择了不同的组件，`MPI_Reduce`的执行方式会完全不同：

- *basic组件*：使用简单的线性收集算法（如上所示）
- *tuned组件*：根据消息大小和进程数量选择二叉树、流水线等优化算法
- *han组件*：使用层次化算法，先在节点内归约，再在节点间归约\ 但对用户而言，调用接口完全相同，这正体现了Open MPI组件架构的优势。]

假设在组件选择过程中选择了basic组件#footnote(note)，调用转入`mca_coll_basic_reduce_log_intra`：

#let code = ```c
// 在ompi/mca/coll/basic/coll_basic_reduce.c中
int mca_coll_basic_reduce_log_intra(const void *sbuf, void *rbuf, 
                                   size_t count,
                                   struct ompi_datatype_t *dtype,
                                   struct ompi_op_t *op,
                                   int root,
                                   struct ompi_communicator_t *comm,
                                   mca_coll_base_module_t *module)
{
    int size = ompi_comm_size(comm);  // 8
    int rank = ompi_comm_rank(comm);
    int dim = comm->c_cube_dim;       // log2(8) = 3
    
    // 检查操作是否可交换
    if (!ompi_op_is_commute(op)) {
        // MPI_SUM是可交换的，所以不会走这个分支
        return ompi_coll_base_reduce_intra_basic_linear(sbuf, rbuf, count, dtype, op, root, comm, module);
    }
    
    // 使用超立方体算法，执行log(N)轮通信
    int vrank = (rank - root + size) % size;  // 虚拟rank，以root为0重新编号
    
    for (int i = 0, mask = 1; i < dim; ++i, mask <<= 1) {
        if (vrank & mask) {
            // 高位进程向低位进程发送并停止
            int peer = ((vrank & ~mask) + root) % size;
            MCA_PML_CALL(send(snd_buffer, count, dtype, peer, 
                             MCA_COLL_BASE_TAG_REDUCE,
                             MCA_PML_BASE_SEND_STANDARD, comm));
            break;
        } else {
            // 低位进程接收并归约
            int peer = vrank | mask;
            if (peer < size) {
                peer = (peer + root) % size;
                MCA_PML_CALL(recv(rcv_buffer, count, dtype, peer,
                                 MCA_COLL_BASE_TAG_REDUCE, comm, 
                                 MPI_STATUS_IGNORE));
                // 执行归约操作
                ompi_op_reduce(op, rcv_buffer, rbuf, count, dtype);
            }
        }
    }
}
```

#figure(
  code,
  caption: [`Basic`组件的`mca_coll_basic_reduce_log_intra`算法]
)

=== 具体执行过程

对于上述对于8进程的reduce操作（root=0），超立方体算法的通信过程如下：

*轮次1（mask=1)*：处理X维度
#show raw.where(block: true): it => sourcecode(numbering: none)[#it]
```c
进程1 → 进程0: 发送数据2，结果：进程0有(1+2)
进程3 → 进程2: 发送数据4，结果：进程2有(3+4)  
进程5 → 进程4: 发送数据6，结果：进程4有(5+6)
进程7 → 进程6: 发送数据8，结果：进程6有(7+8)
```\ #v(-16pt)

*轮次2（mask=2）*：处理Y维度
```c
进程2 → 进程0: 发送(3+4)，结果：进程0有(1+2+3+4)
进程6 → 进程4: 发送(7+8)，结果：进程4有(5+6+7+8)
```\ #v(-16pt)

*轮次3（mask=4）*:处理Z维度
```c
进程4 → 进程0: 发送(5+6+7+8)，结果：进程0有(1+2+3+4+5+6+7+8)=36
```\ #v(-16pt)

#let code = ```c
 *          6----<---7		proc_0: 1+2
 *         /|       /|		proc_1: 2
 *        / |      / |		proc_2: 3+4
 *       /  |     /  |		proc_3: 4
 *      4----<---5   |		proc_4: 5+6
 *      |   2--< |---3		proc_5: 6
 *      |  /     |  /		proc_6: 7+8
 *      | /      | /  		proc_7: 8
 *      |/       |/
 *      0----<---1
```

#figure(
  code,
  caption: [轮次1计算过程示意]
)

#show raw.where(block: true): it => sourcecode()[#it]

=== 调用链总结

完整的`MPI_Reduce`调用链：

#show raw.where(block: true): it => sourcecode(numbering: none)[#it]

```c
用户调用: MPI_Reduce(&local_data, &result, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD)
    ↓
MPI接口: ompi/mpi/c/reduce.c::MPI_Reduce()
    ↓
函数指针分发: comm->c_coll->coll_reduce() [在MPI_Init时设置]
    ↓
组件实现: mca_coll_basic_reduce_intra() [basic组件为例]
    ↓
算法选择: 超立方体Reduce算法 [8进程，基于crossover阈值]
    ↓
底层通信: MCA_PML_CALL(send/recv) [点对点消息传递]
    ↓
结果输出: 进程0获得最终结果36
```
#show raw.where(block: true): it => sourcecode()[#it]

== 主要集合通信算法实现

#let add = [
本章节以Open MPI中的*intra-communicator*（通信子内部）为例进行集合通信算法实现的分析。这些算法用于*单个通信子内部的进程间*集合通信操作，如`MPI_COMM_WORLD`内的所有进程参与的Broadcast、Reduce等操作。

相对的，*inter-communicator*（通信子间）算法用于*两个不同通信子之间*的集合通信，属于更高级的MPI特性，此处不作更多讨论。
]

该部分仅以`Bcast`, `Scatter`, `Gather`,  `Allgather`, `Reduce`为例进行示例性的讨论。 #footnote(add)

=== Bcast

`Bcast`的函数原型如下：

```c
MPI_Bcast(
    void* buffer,
    int count,
    MPI_Datatype datatype,
    int root,
    MPI_Comm communicator)
```\ #v(-16pt)

其中：`buffer`参数在根进程上包含要广播的数据，在其他进程上将接收广播的数据。`count`参数指定数据元素的数量，`datatype`指定数据类型，`root`指定广播的根进程，`communicator`指定参与通信的进程组。

#figure(
  image("../figures/bcast.jpg", width: 50%),
  caption: [MPI_Bcast通信模式图示]
)

Open MPI实现了多种Bcast算法：

==== 线性算法（Linear Algorithm）

*函数*：`ompi_coll_base_bcast_intra_basic_linear()`

源码文件路径：#link("ompi/mca/coll/base/coll_base_bcast.c")

其主要原理是：根进程直接向所有其他进程发送数据。

#let code = ```c
// 根进程使用非阻塞发送向所有其他进程发送数据
if (rank == root) {
    // 分配请求数组
    preq = reqs = ompi_coll_base_comm_get_reqs(module->base_data, size-1);
    // 向所有非根进程发送
    for (i = 0; i < size; ++i) {
        if (i == rank) continue;
        MCA_PML_CALL(isend(buff, count, datatype, i,
                           MCA_COLL_BASE_TAG_BCAST,
                           MCA_PML_BASE_SEND_STANDARD,
                           comm, preq++));
    }
    // 等待所有发送完成
    ompi_request_wait_all(size-1, reqs, MPI_STATUSES_IGNORE);
} else {
    // 非根进程接收数据
    MCA_PML_CALL(recv(buff, count, datatype, root,
                     MCA_COLL_BASE_TAG_BCAST, comm,
                     MPI_STATUS_IGNORE));
}
```

#figure(
  code,
  caption: [代码示例]
)\ #v(-16pt)

图示如下：

#let bcast_linear_diagram = canvas({
  import draw: *
  
  // 绘制根进程
  circle((0, 0), radius: 0.4, fill: rgb("#E8F4F8"), stroke: rgb("#2E86AB") + 1.5pt)
  content((0, 0), text(9pt, weight: "bold")[P0])
  
  // 绘制其他进程
  for i in range(1, 4) {
    let angle = i * 120deg - 30deg
    let x = 2.5 * calc.cos(angle)
    let y = 2.5 * calc.sin(angle)
    
    circle((x, y), radius: 0.4, fill: rgb("#F5F5F5"), stroke: rgb("#666666") + 1pt)
    content((x, y), text(9pt)[P#i])
    
    // 绘制箭头
    line((0.4 * calc.cos(angle), 0.4 * calc.sin(angle)), 
         (x - 0.4 * calc.cos(angle), y - 0.4 * calc.sin(angle)),
         mark: (end: ">"), 
         stroke: rgb("#2E86AB") + 1.5pt)
  }
  
  // 添加数据标识
  content((0, -0.8), text(8pt, fill: rgb("#2E86AB"))[data])
})

#figure(
  bcast_linear_diagram,
  caption: [Broadcast 线性算法图示]
)\ #v(-16pt)

#let add = [
  1. 延迟复杂度（Latency Complexity）\ 定义：算法中通信轮数的度量，表示串行通信步骤的数量\ 表示：$O(p)$ 表示需要 $p$ 轮串行通信 \ 影响因素：网络启动开销 α（每次通信的固定延迟）
  2. 带宽复杂度（Bandwidth Complexity）\ 定义：算法中总的数据传输量\ 表示：$O(p m)$ 表示总共传输 $p m$ 单位的数据\ 影响因素：网络带宽的倒数 $β$（传输单位数据的时间）
  3. 时间复杂度（Time Complexity）\  定义：算法总执行时间的上界估计\  组成：延迟复杂度 + 带宽复杂度 + 计算复杂度，具体的，有：
  $ T_"total" & = T_"latency" + T_"bandwidth" + T_"computation" \
              & = ("通信轮数" dot α) + ("总传输量" dot β) + ("计算时间") $

  \ #v(-16pt)

  而在此处对通信操作复杂度的讨论中，未考虑计算时间的影响
]

算法复杂度分析：线性算法的时间复杂度为$O((p-1) α + (p-1) β m)$，其中$α$为通信启动开销，$β$为带宽的倒数，$p$为进程数，$m$为消息大小。该算法延迟复杂度为$O(p)$，带宽复杂度为$O(p m)$，根进程成为通信瓶颈。#footnote(add)空间复杂度为$O(1)$，无额外空间需求。

#let add = [
  报告中此处的“*适用场景*”为源码注释中提到的经验结果。
]

适用场景#footnote(add)包括小规模通信子($p≤4$)、极小消息大小接近延迟开销的情况、作为复杂算法的回退选择，以及网络连接性差的环境。该算法实现简单且无拓扑构建开销，但根进程瓶颈导致扩展性较差，在大规模或大消息场景下性能显著劣于树形算法。

==== K项树算法（K-nomial Tree Algorithm）

*函数*：`ompi_coll_base_bcast_intra_knomial()`

源码文件路径：#link("ompi/mca/coll/base/coll_base_bcast.c")

#let add = [
  对于给定的radix（分支因子）和进程数量，K项树按以下规则构建：
  1. *根节点*：进程0作为树根
  2. *子节点计算*：对于节点rank，其子节点按公式计算：
     $
     "child_rank" = ("rank" + "size" / "radix" ^ "level" dot "i") mod "size"
     $
     其中$i = 1, 2, ..., min("radix", "remaining_nodes")$
  3. *层次分配*：节点按二进制表示的最高位分组到不同层次
]

按照K-nomial树#footnote(add)结构进行数据传递，根进程作为树根，每个内部节点最多有k个子节点，按照树的层次结构进行数据广播。

#let code = ```c
/*
 * K-nomial tree broadcast algorithm
 * radix参数控制树的分支因子
 */
int ompi_coll_base_bcast_intra_knomial(
    void *buf, size_t count, struct ompi_datatype_t *datatype, int root,
    struct ompi_communicator_t *comm, mca_coll_base_module_t *module,
    uint32_t segsize, int radix)
{
    // 构建k-nomial树
    COLL_BASE_UPDATE_KMTREE(comm, module, root, radix);
    if (NULL == data->cached_kmtree) {
        // 如果构建失败，回退到二项树
        return ompi_coll_base_bcast_intra_binomial(buf, count, datatype, 
                                                   root, comm, module, segcount);
    }
    
    // 使用通用的树形广播算法
    return ompi_coll_base_bcast_intra_generic(buf, count, datatype, root, comm, 
                                              module, segcount, data->cached_kmtree);
}
```

#figure(
  code,
  caption: [K项树Broadcast算法核心代码]
)\ #v(-16pt)

图示如下：

#let knomial_tree_diagram = canvas({
  import draw: *
  
  // 绘制根节点
  circle((0, 0), radius: 0.4, fill: rgb("#E8F4F8"), stroke: rgb("#2E86AB") + 1.5pt)
  content((0, 0), text(9pt, weight: "bold")[P0])
  
  // 第一层子节点 (注释中所示的结构)
  let layer1_x = (-4.5, -2.25, 0, 2.25, 4.5)
  let layer1_y = (-2, -2, -2, -2, -2)
  let layer1_labels = ("P9", "P3", "P6", "P1", "P2")
  
  // 绘制第一层节点
  for i in range(5) {
    circle((layer1_x.at(i), layer1_y.at(i)), radius: 0.4, 
           fill: rgb("#F5F5F5"), stroke: rgb("#666666") + 1pt)
    content((layer1_x.at(i), layer1_y.at(i)), text(9pt)[#layer1_labels.at(i)])
    
    // 从根节点连接到第一层节点
    line((0, -0.4), (layer1_x.at(i), layer1_y.at(i) + 0.4), 
         mark: (end: ">"), stroke: rgb("#2E86AB") + 1.5pt)
  }
  
  // 第二层子节点
  // P3的子节点: P4, P5
  circle((-3, -4), radius: 0.4, fill: rgb("#F5F5F5"), stroke: rgb("#666666") + 1pt)
  content((-3, -4), text(9pt)[P4])
  circle((-1.6, -4), radius: 0.4, fill: rgb("#F5F5F5"), stroke: rgb("#666666") + 1pt)
  content((-1.6, -4), text(9pt)[P5])
  
  // P6的子节点: P7, P8
  circle((0.75, -4), radius: 0.4, fill: rgb("#F5F5F5"), stroke: rgb("#666666") + 1pt)
  content((0.75, -4), text(9pt)[P7])
  circle((-0.65, -4), radius: 0.4, fill: rgb("#F5F5F5"), stroke: rgb("#666666") + 1pt)
  content((-0.65, -4), text(9pt)[P8])
  
  // 从P3连接到其子节点
  line((-2.25, -2.4), (-3, -3.6), mark: (end: ">"), stroke: rgb("#2E86AB") + 1.5pt)
  line((-2.25, -2.4), (-1.6, -3.6), mark: (end: ">"), stroke: rgb("#2E86AB") + 1.5pt)
  
  // 从P6连接到其子节点
  line((0, -2.4), (0.75, -3.6), mark: (end: ">"), stroke: rgb("#2E86AB") + 1.5pt)
  line((0, -2.4), (-0.65, -3.6), mark: (end: ">"), stroke: rgb("#2E86AB") + 1.5pt)
  
  // 添加标题
  content((0, -5.5), text(10pt, fill: rgb("#666666"))[K-nomial Tree (radix=3, comm_size=10)])
})

#figure(
  knomial_tree_diagram,
  caption: [K项树Broadcast算法的树形结构（radix=3）]
)\ #v(-16pt)

算法复杂度分析：K项树算法的时间复杂度为$O(log_"k" (p)α + β m)$，其中`radix`参数$k$控制分支因子。延迟复杂度为$O(log_k (p))$，随着$k$增大而减小，但单节点负载增加；带宽复杂度为$O(m)$，每个消息只传输一次，具有最优的带宽效率。当$k=2$时退化为二叉树，延迟最小；当$k=sqrt(p)$时理论上达到最优权衡。

适用场景包括中大规模通信子($p>8$)、需要调节延迟-带宽权衡的场景，以及层次化网络架构中`radix`可匹配网络拓扑的情况。该算法通过参数化设计在不同网络环境下具有良好的适应性，是Open MPI中重要的可调优广播算法实现。

==== 二叉树广播算法

*函数*：`ompi_coll_base_bcast_intra_bintree`

源码文件路径：#link("ompi/mca/coll/base/coll_base_bcast.c")

使用二叉树结构传播数据，每个节点向两个子节点传递数据。

#let code = ```c
int
ompi_coll_base_bcast_intra_bintree ( void* buffer,
                                      size_t count,
                                      struct ompi_datatype_t* datatype,
                                      int root,
                                      struct ompi_communicator_t* comm,
                                      mca_coll_base_module_t *module,
                                      uint32_t segsize )
{
    size_t segcount = count;
    size_t typelng;
    mca_coll_base_comm_t *data = module->base_data;

    COLL_BASE_UPDATE_BINTREE( comm, module, root );

    /**
     * Determine number of elements sent per operation.
     */
    ompi_datatype_type_size( datatype, &typelng );
    COLL_BASE_COMPUTED_SEGCOUNT( segsize, typelng, segcount );

    OPAL_OUTPUT((ompi_coll_base_framework.framework_output,"……",
                 ompi_comm_rank(comm), segsize, (unsigned long)typelng, segcount));

    return ompi_coll_base_bcast_intra_generic( buffer, count, datatype, root, comm, module,
                                                segcount, data->cached_bintree );
}
```

#figure(
  code,
  caption: [二叉树Broadcast算法核心代码]
)\ #v(-16pt)

#let bintree_diagram = canvas({
  import draw: *
  
  // 绘制根节点
  circle((0, 0), radius: 0.4, fill: rgb("#E8F4F8"), stroke: rgb("#2E86AB") + 1.5pt)
  content((0, 0), text(9pt, weight: "bold")[P0])
  
  // 第一层子节点
  circle((-2, -1.5), radius: 0.4, fill: rgb("#F5F5F5"), stroke: rgb("#666666") + 1pt)
  content((-2, -1.5), text(9pt)[P1])
  
  circle((2, -1.5), radius: 0.4, fill: rgb("#F5F5F5"), stroke: rgb("#666666") + 1pt)
  content((2, -1.5), text(9pt)[P2])
  
  // 第二层子节点
  circle((-3, -3), radius: 0.4, fill: rgb("#F5F5F5"), stroke: rgb("#666666") + 1pt)
  content((-3, -3), text(9pt)[P3])
  
  circle((-1, -3), radius: 0.4, fill: rgb("#F5F5F5"), stroke: rgb("#666666") + 1pt)
  content((-1, -3), text(9pt)[P4])
  
  circle((1, -3), radius: 0.4, fill: rgb("#F5F5F5"), stroke: rgb("#666666") + 1pt)
  content((1, -3), text(9pt)[P5])
  
  circle((3, -3), radius: 0.4, fill: rgb("#F5F5F5"), stroke: rgb("#666666") + 1pt)
  content((3, -3), text(9pt)[P6])
  
  // 第三层节点
  circle((-3.5, -4.5), radius: 0.4, fill: rgb("#F5F5F5"), stroke: rgb("#666666") + 1pt)
  content((-3.5, -4.5), text(9pt)[P7])
  
  // 从根节点连接到第一层
  line((0, -0.4), (-2, -1.1), mark: (end: ">"), stroke: rgb("#2E86AB") + 1.5pt)
  line((0, -0.4), (2, -1.1), mark: (end: ">"), stroke: rgb("#2E86AB") + 1.5pt)
  
  // 从第一层连接到第二层
  line((-2, -1.9), (-3, -2.6), mark: (end: ">"), stroke: rgb("#2E86AB") + 1.5pt)
  line((-2, -1.9), (-1, -2.6), mark: (end: ">"), stroke: rgb("#2E86AB") + 1.5pt)
  line((2, -1.9), (1, -2.6), mark: (end: ">"), stroke: rgb("#2E86AB") + 1.5pt)
  line((2, -1.9), (3, -2.6), mark: (end: ">"), stroke: rgb("#2E86AB") + 1.5pt)
  
  // 从第二层连接到第三层
  line((-3, -3.4), (-3.5, -4.1), mark: (end: ">"), stroke: rgb("#2E86AB") + 1.5pt)
  
  // 添加说明
  content((0, -5.5), text(10pt, fill: rgb("#666666"))[Binary Tree (8进程)]
  )
})

#figure(
  bintree_diagram,
  caption: [二叉树Broadcast算法的树形结构]
)\ #v(-16pt)

复杂度分析：二叉树广播算法的时间复杂度为$O(log_2(p) α + β m)$，延迟复杂度为$O(log p)$，带宽复杂度为$O(m)$。相比线性算法，通信轮数从$O(p)$降低到$O(log p)$，显著减少了延迟开销。该算法支持消息分段处理，通过`segsize`参数控制分段大小，在大消息传输时能够提高内存利用效率。

适用场景包括中等规模通信子($4 <= p <= 64$)、延迟敏感应用、中等大小消息($1"KB"-1"MB"$)，以及进程数为$2$的幂次时性能最优的情况。二叉树结构在延迟和实现复杂度之间达到良好平衡，是许多MPI实现中的默认选择，特别适合CPU密集型应用中的小到中等规模数据广播。

==== 流水线广播算法（Pipeline Algorithm）

*函数*：`ompi_coll_base_bcast_intra_pipeline()`

源码文件路径：#link("ompi/mca/coll/base/coll_base_bcast.c")

其主要原理是：将大消息分割成多个小段（segments），在线性链结构上采用流水线方式传递数据，使不同数据段的传输可以重叠进行，提高带宽利用率。

#let code = ```c
int
ompi_coll_base_bcast_intra_pipeline( void* buffer,
                                      size_t count,
                                      struct ompi_datatype_t* datatype,
                                      int root,
                                      struct ompi_communicator_t* comm,
                                      mca_coll_base_module_t *module,
                                      uint32_t segsize )
{
    size_t segcount = count;
    size_t typelng;
    mca_coll_base_comm_t *data = module->base_data;

    COLL_BASE_UPDATE_PIPELINE( comm, module, root );

    /**
     * Determine number of elements sent per operation.
     */
    ompi_datatype_type_size( datatype, &typelng );
    COLL_BASE_COMPUTED_SEGCOUNT( segsize, typelng, segcount );

    OPAL_OUTPUT((ompi_coll_base_framework.framework_output,......

    return ompi_coll_base_bcast_intra_generic( buffer, count, datatype, root, comm, module,
                                                segcount, data->cached_pipeline );
}
```

#figure(
  code,
  caption: [流水线Broadcast算法核心代码]
)\ #v(-16pt)

图示如下：

#let pipeline_bcast_diagram = canvas({
  import draw: *
  
  // 绘制时间轴
  line((1, 0), (8, 0), stroke: gray + 1pt)
  content((4.5, -0.5), text(9pt)[时间 →])
  
  // 绘制进程节点和时间线
  for i in range(4) {
    let y = i * 1.5 + 1
    
    // 进程标签
    circle((0.5, y), radius: 0.3, 
           fill: if i == 0 { rgb("#E8F4F8") } else { rgb("#F5F5F5") },
           stroke: if i == 0 { rgb("#2E86AB") + 1.5pt } else { rgb("#666666") + 1pt })
    content((0.5, y), text(8pt, weight: if i == 0 { "bold" } else { "regular" })[P#i])
    
    // 进程时间线
    line((1, y), (8, y), stroke: gray.lighten(50%) + 0.5pt)
  }
  
  // 时间点标记
  for t in range(1, 6) {
    let x = t * 1.2 + 1
    line((x, 0.8), (x, 5.2), stroke: gray.lighten(70%) + 0.5pt)
    content((x, 0.5), text(8pt)[t#t])
  }
  
  // 数据段颜色
  let colors = (rgb("#E74C3C"), rgb("#3498DB"), rgb("#2ECC71"))
  
  // 数据段1的流动
  // t1: 段1在P0
  rect((2.2, 0.8), (3.4, 1.2), fill: colors.at(0).lighten(70%), stroke: colors.at(0) + 1pt)
  content((2.8, 1), text(7pt, fill: colors.at(0))[段1])
  
  // t2: 段1到P1，段2在P0
  rect((3.4, 2.3), (4.6, 2.7), fill: colors.at(0).lighten(70%), stroke: colors.at(0) + 1pt)
  content((4, 2.5), text(7pt, fill: colors.at(0))[段1])
  rect((3.4, 0.8), (4.6, 1.2), fill: colors.at(1).lighten(70%), stroke: colors.at(1) + 1pt)
  content((4, 1), text(7pt, fill: colors.at(1))[段2])
  
  // t3: 段1到P2，段2到P1，段3在P0
  rect((4.6, 3.8), (5.8, 4.2), fill: colors.at(0).lighten(70%), stroke: colors.at(0) + 1pt)
  content((5.2, 4), text(7pt, fill: colors.at(0))[段1])
  rect((4.6, 2.3), (5.8, 2.7), fill: colors.at(1).lighten(70%), stroke: colors.at(1) + 1pt)
  content((5.2, 2.5), text(7pt, fill: colors.at(1))[段2])
  rect((4.6, 0.8), (5.8, 1.2), fill: colors.at(2).lighten(70%), stroke: colors.at(2) + 1pt)
  content((5.2, 1), text(7pt, fill: colors.at(2))[段3])
  
  // 绘制传输箭头
  line((3.4, 1), (3.4, 2.5), mark: (end: ">"), stroke: colors.at(0) + 1.5pt)
  line((4.6, 2.5), (4.6, 4), mark: (end: ">"), stroke: colors.at(0) + 1.5pt)
  line((4.6, 1), (4.6, 2.5), mark: (end: ">"), stroke: colors.at(1) + 1.5pt)
  
  // 说明文字
  content((7, 3), [
    #set align(left)
    #text(8pt)[
      流水： \
      • 数据分段传输 \
      • 并行处理多段 \
      • 提高带宽利用率
    ]
  ])
})

#figure(
  pipeline_bcast_diagram,
  caption: [流水线Broadcast算法图示]
)\ #v(-16pt)

算法复杂度分析：流水线广播算法的时间复杂度为$O((log_2(p) + S-1)α + β m)$，其中$S$为段数。通过消息分割和流水线重叠，延迟复杂度为$O(log p + S)$，带宽复杂度保持$O(m)$但具有更好的带宽利用率。分段大小(`segsize`)直接影响性能：较小分段提供更好的重叠效果但增加通信开销，较大分段减少开销但降低重叠效益。

适用场景包括大消息广播($>1"MB"$)、带宽充足但延迟较高的网络环境、内存受限环境中分段可减少内存压力，以及需要通信-计算重叠的应用。该算法通过流水线技术有效隐藏通信延迟，在高性能计算中的大规模数据分发场景下表现优异，是带宽密集型应用的理想选择。

==== 分散-聚集广播算法

*函数*：`ompi_coll_base_bcast_intra_scatter_allgather`

源码文件路径：#link("ompi/mca/coll/base/coll_base_bcast.c")

先使用二项树分散数据，再使用递归倍增方式聚集。

其主要原理是：采用两阶段策略，第一阶段使用二项树将数据分散到各进程（Scatter），第二阶段使用递归倍增算法进行全聚集（Allgather），重构完整数据。

#let code = ```c
/*
 * 限制条件: count >= comm_size
 */
int ompi_coll_base_bcast_intra_scatter_allgather(
    void *buf, size_t count, struct ompi_datatype_t *datatype, int root,
    struct ompi_communicator_t *comm, mca_coll_base_module_t *module,
    uint32_t segsize)
{
    int comm_size = ompi_comm_size(comm);
    int rank = ompi_comm_rank(comm);
    int vrank = (rank - root + comm_size) % comm_size;
    
    // 计算每个进程应分得的数据块大小
    size_t scatter_count = (count + comm_size - 1) / comm_size;
    
    /* 第一阶段：二项树分散 */
    int mask = 0x1;
    while (mask < comm_size) {
        if (vrank & mask) {
            // 从父进程接收数据
            int parent = (rank - mask + comm_size) % comm_size;
            recv_count = rectify_diff(count, vrank * scatter_count);
            MCA_PML_CALL(recv((char *)buf + vrank * scatter_count * extent,
                             recv_count, datatype, parent,
                             MCA_COLL_BASE_TAG_BCAST, comm, &status));
            break;
        }
        mask <<= 1;
    }
    
    // 向子进程发送数据
    mask >>= 1;
    while (mask > 0) {
        if (vrank + mask < comm_size) {
            int child = (rank + mask) % comm_size;
            send_count = rectify_diff(curr_count, scatter_count * mask);
            MCA_PML_CALL(send((char *)buf + scatter_count * (vrank + mask) * extent,
                             send_count, datatype, child,
                             MCA_COLL_BASE_TAG_BCAST,
                             MCA_PML_BASE_SEND_STANDARD, comm));
        }
        mask >>= 1;
    }
    
    /* 第二阶段：递归倍增全聚集 */
    mask = 0x1;
    while (mask < comm_size) {
        int vremote = vrank ^ mask;
        int remote = (vremote + root) % comm_size;
        
        if (vremote < comm_size) {
            // 与远程进程交换数据
            ompi_coll_base_sendrecv((char *)buf + send_offset,
                                    curr_count, datatype, remote,
                                    MCA_COLL_BASE_TAG_BCAST,
                                    (char *)buf + recv_offset,
                                    recv_count, datatype, remote,
                                    MCA_COLL_BASE_TAG_BCAST,
                                    comm, &status, rank);
            curr_count += recv_count;
        }
        mask <<= 1;
    }
    
    return MPI_SUCCESS;
}
```

#figure(
  code,
  caption: [分散-聚集Broadcast算法核心代码]
)\ #v(-16pt)

图示如下：

#let scatter_allgather_diagram = canvas({
  import draw: *
  
  // 整体布局调整：左右分布，更清晰的间距
  
  // ===== 第一阶段：Scatter =====
  content((-3, 3), text(11pt, weight: "bold", fill: rgb("#2E86AB"))[阶段1: 二项树分散])
  
  // 绘制根进程
  circle((-3, 1.5), radius: 0.4, fill: rgb("#E8F4F8"), stroke: rgb("#2E86AB") + 1.5pt)
  content((-3, 1.5), text(9pt, weight: "bold")[P0])
  content((-3, 0.8), text(7pt, fill: rgb("#2E86AB"))[完整数据])
  
  // 第一层分散 - 调整位置，更整齐
  circle((-5, 0), radius: 0.4, fill: rgb("#F5F5F5"), stroke: rgb("#666666") + 1pt)
  content((-5, 0), text(9pt)[P4])
  content((-5, -0.6), text(7pt, fill: rgb("#E74C3C"))[D0D1])
  
  circle((-1, 0), radius: 0.4, fill: rgb("#F5F5F5"), stroke: rgb("#666666") + 1pt)
  content((-1, 0), text(9pt)[P2])
  content((-1, -0.6), text(7pt, fill: rgb("#3498DB"))[D2D3])
  
  // 第二层分散 - 水平对齐
  circle((-6, -1.8), radius: 0.4, fill: rgb("#F5F5F5"), stroke: rgb("#666666") + 1pt)
  content((-6, -1.8), text(9pt)[P6])
  content((-6, -2.4), text(7pt, fill: rgb("#E74C3C"))[D0])
  
  circle((-4, -1.8), radius: 0.4, fill: rgb("#F5F5F5"), stroke: rgb("#666666") + 1pt)
  content((-4, -1.8), text(9pt)[P5])
  content((-4, -2.4), text(7pt, fill: rgb("#E74C3C"))[D1])
  
  circle((-2, -1.8), radius: 0.4, fill: rgb("#F5F5F5"), stroke: rgb("#666666") + 1pt)
  content((-2, -1.8), text(9pt)[P3])
  content((-2, -2.4), text(7pt, fill: rgb("#3498DB"))[D2])
  
  circle((0, -1.8), radius: 0.4, fill: rgb("#F5F5F5"), stroke: rgb("#666666") + 1pt)
  content((0, -1.8), text(9pt)[P1])
  content((0, -2.4), text(7pt, fill: rgb("#3498DB"))[D3])
  
  // 分散阶段的箭头 - 更清晰的路径
  line((-3, 1.1), (-5, 0.4), mark: (end: ">"), stroke: rgb("#E74C3C") + 1.5pt)
  line((-3, 1.1), (-1, 0.4), mark: (end: ">"), stroke: rgb("#3498DB") + 1.5pt)
  line((-5, -0.4), (-6, -1.4), mark: (end: ">"), stroke: rgb("#E74C3C") + 1.5pt)
  line((-5, -0.4), (-4, -1.4), mark: (end: ">"), stroke: rgb("#E74C3C") + 1.5pt)
  line((-1, -0.4), (-2, -1.4), mark: (end: ">"), stroke: rgb("#3498DB") + 1.5pt)
  line((-1, -0.4), (0, -1.4), mark: (end: ">"), stroke: rgb("#3498DB") + 1.5pt)
  
  // ===== 第二阶段：Allgather =====
  content((4, 3), text(11pt, weight: "bold", fill: rgb("#2E86AB"))[阶段2: 递归倍增聚集])
  
  // 统一的进程排列 - 8个进程水平排列
  let proc_base_x = 2
  let proc_y = 2
  
  for i in range(8) {
    let x = proc_base_x + i * 0.7
    circle((x, proc_y), radius: 0.25, fill: rgb("#F5F5F5"), stroke: rgb("#666666") + 1pt)
    content((x, proc_y), text(7pt)[P#i])
  }
  
  let leftbase = 3
  let overbase = -1

  // 轮次1: 距离1交换 - 修复range()语法
  content((leftbase - 0.15, overbase + 1.3), text(9pt, fill: rgb("#666666"))[轮次1: 相邻交换])

  content((7, overbase + 1.3), text(8pt, fill: rgb("#666666"))[*轮次1后*：每进程2个数据块])
  
  // 绘制距离1的交换箭头 - 使用while循环代替range
  let i = 0
  while i < 8 {
    let x1 = proc_base_x + i * 0.7
    let x2 = proc_base_x + (i + 1) * 0.7
    line((x1, proc_y - 0.5), (x2, proc_y - 0.5), 
         mark: (end: ">", start: ">"), stroke: rgb("#E74C3C") + 1pt)
    i = i + 2
  }
  
  // 轮次2: 距离2交换
  content((leftbase, overbase + 0.6), text(9pt, fill: rgb("#666666"))[轮次2: 距离2交换])

  content((7, overbase + 0.6), text(8pt, fill: rgb("#666666"))[*轮次2后*：每进程4个数据块])
  
  // 绘制距离2的交换箭头
  let j = 0
  while j < 8 {
    let x1 = proc_base_x + j * 0.7
    let x2 = proc_base_x + (j + 2) * 0.7
    line((x1, proc_y - 0.8), (x2, proc_y - 0.8), 
         mark: (end: ">", start: ">"), stroke: rgb("#3498DB") + 1pt)
    j = j + 4
  }
  
  // 轮次3: 距离4交换
  content((leftbase, overbase - 0.1), text(9pt, fill: rgb("#666666"))[轮次3: 距离4交换])

  content((7, overbase - 0.1), text(8pt, fill: rgb("#666666"))[*轮次3后*：每进程8个数据块])
  
  
  // 绘制距离4的交换箭头
  let x1 = proc_base_x
  let x2 = proc_base_x + 4 * 0.7
  line((x1, proc_y - 1.1), (x2, proc_y - 1.1), 
       mark: (end: ">", start: ">"), stroke: rgb("#2ECC71") + 1pt)
  
  // // 数据状态说明 - 右侧整齐排列
  // content((7.5, overbase + 0.7), [
  //   #set align(left)
  //   #text(8pt, fill: rgb("#666666"))[
      
  //     *初始*：每进程1个数据块 \
  //     *轮次1后*：每进程2个数据块 \
  //     *轮次2后*：每进程4个数据块 \
  //     *轮次3后*：每进程8个数据块 \
  //     \
  //     *结果*：所有进程获得完整数据
  //   ]
  // ])
  
  // 添加分隔线
  line((-7, -3.5), (10, -3.5), stroke: gray.lighten(50%) + 0.5pt)
  
  // 底部总结
  content((1.5, -4), text(10pt, fill: rgb("#666666"), weight: "bold")[
    分散-聚集广播算法：8进程，3轮递归倍增聚集
  ])
})

#figure(
  scatter_allgather_diagram,
  caption: [分散-聚集Broadcast算法图示]
)\ #v(-16pt)

算法复杂度分析：分散-聚集广播算法的时间复杂度为$O(2log_2(p)α + 2β m(p-1)/p)$，包含两个阶段：二项树分散阶段$O(log_2(p)α + β m(p-1)/p)$和递归倍增聚集阶段$O(log_2 (p)α + β m(p-1)/p)$。总延迟复杂度为$O(log p)$，总带宽复杂度为$O(m(p-1)/p)$，当$p$较大时接近$O(m)$的最优带宽效率。该算法要求$"count"≥"comm_size"$，当消息过小时会回退到线性算法。

适用场景包括大消息广播($"count"≥"comm_size"$)、大规模通信子($p>64$)、高带宽网络环境，以及需要避免根节点瓶颈的场景。通过两阶段设计，该算法充分利用聚合带宽并避免单点瓶颈，在大消息和大规模场景下具有近似线性的带宽效率，是高性能计算中处理大规模数据广播的重要算法选择。

==== 其它Broadcast算法

在源码#link("ompi/mca/coll/base/coll_base_bcast.c")中，除了上述详细介绍的算法外，还实现了以下其它Broadcast算法：

#list(
[链式广播算法（ompi_coll_base_bcast_intra_chain）\
形成一个或多个通信链，数据沿链传递。支持通过fanout参数控制多链并行，适合特定网络拓扑结构。],

[分裂二进制树算法（ompi_coll_base_bcast_intra_split_bintree）\
将树结构和数据进行分割以优化传输效率，通过更复杂的调度在某些场景下实现更高的性能。],

[分散-环形聚集算法（ompi_coll_base_bcast_intra_scatter_allgather_ring）\
结合二项树分散和环形聚集的混合策略，先使用二项树分散数据，再使用环形算法进行聚集，在特定网络拓扑上更高效。],

[通用树形算法（ompi_coll_base_bcast_intra_generic）\
提供通用的树形广播框架，可以配合不同的树结构（二叉树、k进制树等）实现灵活的广播策略。] )\ #v(-16pt)

这些算法的设计目标是适应不同的通信规模、消息大小和网络特性。Open MPI的动态选择机制会根据运行时条件（进程数量、消息大小、网络延迟等）自动选择最优的算法实现，为用户提供透明的性能优化。


==== 总结

基于上述对`MPI_Bcast`的算法的讨论，整理得如下表格：

#let summary = table(
  columns: (0.9fr, 2fr, 0.9fr, 2fr, 1.5fr),
  align: (left, left, left, left, left),
  stroke: 0.5pt,
  table.header(
    [*算法名称*], 
    [*函数名称*], 
    [*可选参数*], 
    [*时间复杂度*], 
    [*适用场景*]
  ),
  
  [线性算法], 
  [#text(size: 8pt)[`ompi_coll_base_bcast_`\ `intra_basic_linear`]], 
  [无], 
  [$O(N α + N β m)$], 
  [小规模通信子\ 或回退选择],
  
  [二叉树算法], 
  [#text(size: 8pt)[`ompi_coll_base_bcast_`\ `intra_bintree`]], 
  [`segsize`], 
  [$O(log_2(p) α + β m)$], 
  [中等规模通信子\ 延迟敏感应用],
  
  [二项式树算法], 
  [#text(size: 8pt)[`ompi_coll_base_bcast_`\ `intra_binomial`]], 
  [`segsize`], 
  [$O(log_2(p)α + β m)$], 
  [中等规模通信子\ 支持消息分段],
  
  [K项树算法], 
  [#text(size: 8pt)[`ompi_coll_base_bcast_`\ `intra_knomial`]], 
  [#text(size: 8pt)[`segsize`\ `radix`]], 
  [$O(log_k(p)α + β m)$], 
  [可调节延迟-带宽\ 权衡的中大规模通信],
  
  [流水线算法], 
  [#text(size: 8pt)[`ompi_coll_base_bcast_`\ `intra_pipeline`]], 
  [`segsize`], 
  [$O((log_2(p) + S)α + β m)$], 
  [大消息广播\ 通信-计算重叠],
  
  [链式算法], 
  [#text(size: 8pt)[`ompi_coll_base_bcast_`\ `intra_chain`]], 
  [#text(size: 8pt)[`segsize`\ `chains`]], 
  [$O(N/"chains" dot α + β m)$], 
  [特定网络拓扑\ 多链并行传输],
  
  [分散-聚集算法], 
  [#text(size: 8pt)[`ompi_coll_base_bcast_`\
  `intra_scatter_allgather`]], 
  [`segsize`], 
  [$O(α log p + β m)$], 
  [大消息广播\ 避免根节点瓶颈],
  
  [分散-环形聚集算法], 
  [#text(size: 8pt)[`ompi_coll_base_bcast_`\ `intra_scatter_allgather_ring`]], 
  [`segsize`], 
  [$O(α(log(p) + p) + β m)$], 
  [超大规模通信子\ 带宽受限网络],
  
  [分裂二叉树算法], 
  [#text(size: 8pt)[`ompi_coll_base_bcast_`\ `intra_split_bintree`]], 
  [`segsize`], 
  [$O(log₂(p)α + β m)$], 
  [数据和树结构\ 分割优化场景],
  
  [通用树形算法], 
  [#text(size: 8pt)[`ompi_coll_base_bcast_`\ `intra_generic`]], 
  [#text(size: 8pt)[`tree`\ `segcount`]], 
  [取决于树结构], 
  [通用框架\ 配合不同树结构]
)

#summary

#align(center)[
  #text[表 3.1：Open MPI Broadcast算法总结]
]

#align(left)[
  #columns(2)[
    *参数说明：*
    - $S$: 流水线算法中的段数
    - $α$: 通信延迟参数，$β$: 带宽倒数参数
    - $m$: 消息大小，$p$: 进程数量
    - `segsize`: 控制消息分段大小的参数
    - `radix`: K项树的分支因子（$≥2$）
    
    #colbreak()
    \
    - `chains`: 链式算法中并行链的数量
    - `tree`: 指定使用的树结构类型
    - `segcount`: 每段传输的元素数量
  ]
]

=== Scatter

`Scatter`的函数原型如下:

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
```

\ #v(-16pt)

其中：`send_data`参数是只在根进程上有效的待分发数据数组。`recv_data`是所有进程接收数据的缓冲区。`send_count`和`recv_count`分别指定发送和接收的数据元素数量。

#figure(
  image("../figures/scatter.jpg", width: 50%),
  caption: [MPI_Scatter通信模式图示]
)

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

算法复杂度分析：二项式树散射算法的时间复杂度为$O(alpha log(p) + β m(p-1)/p)$，其中$m = "scount" × "comm_size"$为总数据量。延迟复杂度为$O(log p)$，相比线性算法的$O(p)$有显著改善；带宽复杂度为$O(m(p-1)/p)$，当进程数较大时接近$O(m)$的最优效率。算法内存需求因角色而异：根进程需要$"scount"  "comm_size" times "sdtype_size"$内存，非根非叶进程需要$"rcount" times "comm_size" times "rdtype_size"$内存。

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

算法复杂度分析：线性散射算法的时间复杂度为$O((p-1)α + (p-1)β m^')$，其中$m^' = "scount"$为单个数据块大小。延迟复杂度为$O(p)$，根进程需要进行$p-1$次串行发送操作；带宽复杂度为$O(p m^')$，总传输量为所有数据块之和。该算法实现最为简单#footnote(add)，无需构建树形拓扑，空间复杂度为$O(1)$。

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

算法复杂度分析：非阻塞线性散射算法的时间复杂度与标准线性算法相同，为$O((p-1)α + (p-1)β m')$，但通过非阻塞通信获得更好的重叠效果。延迟复杂度理论上仍为$O(p)$，但实际延迟因通信重叠而降低；带宽复杂度为$O(p m')$。该算法通过`max_reqs`参数#footnote(add)控制资源使用，在内存需求和性能之间提供可调节的权衡。

适用场景包括中等规模通信子、需要通信-计算重叠的应用、内存资源有限但希望改善性能的环境，以及网络具有良好并发处理能力的场景。该算法在保持线性算法简单性的同时，通过非阻塞技术提升了性能，是资源受限环境下的良好选择。

==== 其它Scatter算法

除了上述实现的算法外，base组件中的的Scatter操作中还还包含采用以下算法和优化策略：

#list(
[链式散射算法（ompi_coll_base_scatter_intra_chain）\
形成一个或多个通信链，数据沿链传递，适合特定网络拓扑结构。],

[分段流水线算法（ompi_coll_base_scatter_intra_pipeline）\
将大消息分段，采用流水线方式在链式或树形结构上传递，提升大消息分发效率。],

[通用树形算法（ompi_coll_base_scatter_intra_generic）\
提供通用的树形分发框架，可配合不同树结构（如二叉树、k进制树等）实现灵活策略。],

[基于网络拓扑的优化算法（如ompi_coll_base_scatter_intra_topo）\
根据具体网络拓扑（如胖树、环面等）优化数据分发路径，减少拥塞。],

[混合算法策略\
根据消息大小和进程数量动态选择算法，小消息用线性，大消息用树形或链式算法。],
)\ #v(-16pt)


==== 总结

基于上述对`MPI_Scatter`的算法的讨论，整理得如下表格：

#let scatter_summary = table(
  columns: (0.9fr, 2.2fr, 0.9fr, 2.3fr, 1.5fr),
  align: (left, left, left, left, left),
  stroke: 0.5pt,
  table.header(
    [*算法名称*], 
    [*函数名称*], 
    [*可选参数*], 
    [*时间复杂度*], 
    [*适用场景*]
  ),
  
  [二项式树算法], 
  [#text(size: 8pt)[`ompi_coll_base_scatter_`\ `intra_binomial`]], 
  [无], 
  [$O(α log(p) + β m(p-1)/p)$], 
  [大规模通信子\ 大消息分发],
  
  [线性算法], 
  [#text(size: 8pt)[`ompi_coll_base_scatter_`\ `intra_basic_linear`]], 
  [无], 
  [$O((p-1)α + (p-1)β m')$], 
  [小规模通信子\ 或回退选择],
  
  [非阻塞线性算法], 
  [#text(size: 8pt)[`ompi_coll_base_scatter_`\ `intra_linear_nb`]], 
  [`max_reqs`], 
  [$O((p-1)α + (p-1)β m')$], 
  [中等规模通信子\ 通信-计算重叠],
  
  [链式散射算法], 
  [#text(size: 8pt)[`ompi_coll_base_scatter_`\ `intra_chain`]], 
  [#text(size: 8pt)[`segsize`\ `chains`]], 
  [$O(p/"chains" dot α + β m)$], 
  [特定网络拓扑\ 多链并行传输],
  
  [分段流水线算法], 
  [#text(size: 8pt)[`ompi_coll_base_scatter_`\ `intra_pipeline`]], 
  [`segsize`], 
  [$O((log_2(p) + S)α + β m)$], 
  [大消息分发\ 流水线重叠],
  
  [通用树形算法], 
  [#text(size: 8pt)[`ompi_coll_base_scatter_`\ `intra_generic`]], 
  [#text(size: 8pt)[`tree`\ `segcount`]], 
  [取决于树结构], 
  [通用框架\ 配合不同树结构],
  
  [基于拓扑的算法], 
  [#text(size: 8pt)[`ompi_coll_base_scatter_`\ `intra_topo`]], 
  [`topology`], 
  [取决于网络拓扑], 
  [特定网络架构\ 拓扑感知优化],
)

#scatter_summary

#align(center)[
  #text[表 3.2：Open MPI Scatter算法总结]
]

#align(left)[
  #columns(2)[
    *参数说明：*
    - $S$: 流水线算法中的段数
    - $α$: 通信延迟参数，$β$: 带宽倒数参数
    - $m$: 总消息大小（scount × comm_size），$m'$: 单个数据块大小（scount）
    - $p$: 进程数量
    - `max_reqs`: 控制非阻塞发送请求数量
    
    #colbreak()
    \
    - `chains`: 链式算法中并行链的数量
    - `tree`: 指定使用的树结构类型
    - `segcount`: 每段传输的元素数量
    - `segsize`: 控制消息分段大小的参数
    - `topology`: 网络拓扑结构参数
  ]
]

=== Gather

`Gather`的函数原型如下：

```c
MPI_Gather(
    void* send_data,
    int send_count,
    MPI_Datatype send_type,
    void* recv_data,
    int recv_count,
    MPI_Datatype recv_type,
    int root,
    MPI_Comm communicator)
```

\ #v(-16pt)

其中：`send_data`是每个进程要发送的数据，`recv_data`是根进程接收所有数据的缓冲区（仅在根进程有效）。`send_count`和`recv_count`分别指定每个进程发送和根进程从每个进程接收的数据元素数量。

#figure(
  image("../figures/gather.png", width: 50%),
  caption: [MPI_Gather通信模式图示]
)

Open MPI为Gather操作提供了多种算法实现：

==== 二项式树算法（Binomial Tree Algorithm）

*函数*：`ompi_coll_base_gather_intra_binomial()`

源码文件路径：#link("ompi/mca/coll/base/coll_base_gather.c")

其主要原理是：使用二项式树结构从叶节点向根节点收集数据，每个内部节点先从子节点收集数据，然后将收集到的数据连同自己的数据一起发送给父节点，最终所有数据汇聚到根进程。

#let code = ```c
int
ompi_coll_base_gather_intra_binomial(const void *sbuf, size_t scount,
                                      struct ompi_datatype_t *sdtype,
                                      void *rbuf, size_t rcount,
                                      struct ompi_datatype_t *rdtype,
                                      int root,
                                      struct ompi_communicator_t *comm,
                                      mca_coll_base_module_t *module)
{
    // 创建二项式树
    COLL_BASE_UPDATE_IN_ORDER_BMTREE(comm, base_module, root);
    bmtree = data->cached_in_order_bmtree;
    
    vrank = (rank - root + size) % size;
    
    if (rank == root) {
        // 根进程：分配接收缓冲区
        if (0 == root) {
            ptmp = (char *) rbuf;
            if (sbuf != MPI_IN_PLACE) {
                err = ompi_datatype_sndrcv((void *)sbuf, scount, sdtype,
                                           ptmp, rcount, rdtype);
            }
        } else {
            // 非0根进程需要额外缓冲区，最后进行数据旋转
            tempbuf = (char *) malloc(rsize);
            ptmp = tempbuf - rgap;
            if (sbuf != MPI_IN_PLACE) {
                err = ompi_datatype_sndrcv((void *)sbuf, scount, sdtype,
                                           ptmp, rcount, rdtype);
            }
        }
        total_recv = rcount;
    } else if (!(vrank % 2)) {
        // 内部节点：分配临时缓冲区用于收集子节点数据
        tempbuf = (char *) malloc(ssize);
        ptmp = tempbuf - sgap;
        // 复制本地数据到临时缓冲区
        err = ompi_datatype_sndrcv((void *)sbuf, scount, sdtype,
                                   ptmp, scount, sdtype);
        total_recv = rcount;
    } else {
        // 叶节点：直接使用发送缓冲区
        ptmp = (char *) sbuf;
        total_recv = scount;
    }
    
    if (!(vrank % 2)) {
        // 所有非叶节点从子节点接收数据
        for (i = 0; i < bmtree->tree_nextsize; i++) {
            int mycount = 0, vkid;
            vkid = (bmtree->tree_next[i] - root + size) % size;
            mycount = vkid - vrank;
            if (mycount > (size - vkid))
                mycount = size - vkid;
            mycount *= rcount;
            
            err = MCA_PML_CALL(recv(ptmp + total_recv*rextent, 
                                    (ptrdiff_t)rcount * size - total_recv, rdtype,
                                    bmtree->tree_next[i], MCA_COLL_BASE_TAG_GATHER,
                                    comm, &status));
            total_recv += mycount;
        }
    }
    
    if (rank != root) {
        // 所有非根节点向父节点发送数据
        err = MCA_PML_CALL(send(ptmp, total_recv, sdtype,
                                bmtree->tree_prev,
                                MCA_COLL_BASE_TAG_GATHER,
                                MCA_PML_BASE_SEND_STANDARD, comm));
    }
    
    if (rank == root && root != 0) {
        // 非0根进程需要进行数据旋转
        err = ompi_datatype_copy_content_same_ddt(rdtype, 
                                (ptrdiff_t)rcount * (ptrdiff_t)(size - root),
                                (char *)rbuf + rextent * (ptrdiff_t)root * (ptrdiff_t)rcount, 
                                ptmp);
        
        err = ompi_datatype_copy_content_same_ddt(rdtype, 
                                (ptrdiff_t)rcount * (ptrdiff_t)root,
                                (char *) rbuf, 
                                ptmp + rextent * (ptrdiff_t)rcount * (ptrdiff_t)(size-root));
        free(tempbuf);
    }
    
    return MPI_SUCCESS;
}
```

#figure(
  code,
  caption: [二项式树Gather算法核心代码]
)\ #v(-16pt)

#let add = [
  二项式树gather的内存需求分析：
  1. *根进程*：需要$"rcount" × "comm_size" × "rdtype_size"$内存存储最终结果
  2. *内部节点*：需要$"scount" × "comm_size" × "sdtype_size"$内存作为临时缓冲区
  3. *叶节点*：仅需要自身数据大小，无额外内存开销
  4. *非0根进程*：额外需要临时缓冲区用于数据旋转操作
]

算法复杂度分析：二项式树聚集算法的时间复杂度为$O(alpha log(p) + β m'(p-1))$，其中$m' = "scount"$为单个进程的数据大小。延迟复杂度为$O(log p)$，相比线性算法的$O(p)$有显著改善；带宽复杂度为$O(m'p)$，总传输量为所有进程数据之和。该算法内存需求#footnote(add)因节点角色而异，通过树形结构有效减少了通信轮数。

适用场景包括大规模通信子($p>8$)、大消息收集、延迟敏感应用，以及需要减少通信轮数的场景。该算法通过二项式树结构在通信轮数和实现复杂度之间达到良好平衡，特别适合高性能计算中需要高效数据收集的应用场景。

==== 线性同步算法（Linear Sync Algorithm）

*函数*：`ompi_coll_base_gather_intra_linear_sync()`

源码文件路径：#link("ompi/mca/coll/base/coll_base_gather.c")

其主要原理是：增加同步机制的线性收集算法，根进程首先向非根进程发送零字节同步消息，然后非根进程分两阶段发送数据：先发送第一段数据（同步），再发送剩余数据，确保数据传输的有序性和可靠性。

#let code = ```c
int
ompi_coll_base_gather_intra_linear_sync(const void *sbuf, size_t scount,
                                         struct ompi_datatype_t *sdtype,
                                         void *rbuf, size_t rcount,
                                         struct ompi_datatype_t *rdtype,
                                         int root,
                                         struct ompi_communicator_t *comm,
                                         mca_coll_base_module_t *module,
                                         int first_segment_size)
{
    if (rank != root) {
        // 非根进程：三步骤通信
        // 1. 接收根进程的零字节同步消息
        ret = MCA_PML_CALL(recv(rbuf, 0, MPI_BYTE, root,
                                MCA_COLL_BASE_TAG_GATHER,
                                comm, MPI_STATUS_IGNORE));
        
        // 2. 同步发送第一段数据
        ompi_datatype_type_size(sdtype, &typelng);
        ompi_datatype_get_extent(sdtype, &lb, &extent);
        first_segment_count = scount;
        COLL_BASE_COMPUTED_SEGCOUNT((size_t)first_segment_size, typelng,
                                    first_segment_count);
        
        ret = MCA_PML_CALL(send(sbuf, first_segment_count, sdtype, root,
                                MCA_COLL_BASE_TAG_GATHER,
                                MCA_PML_BASE_SEND_STANDARD, comm));
        
        // 3. 发送剩余数据
        ret = MCA_PML_CALL(send((char*)sbuf + extent * first_segment_count,
                                (scount - first_segment_count), sdtype,
                                root, MCA_COLL_BASE_TAG_GATHER,
                                MCA_PML_BASE_SEND_STANDARD, comm));
        
    } else {
        // 根进程：与每个非根进程进行复杂的同步通信
        for (i = 0; i < size; ++i) {
            if (i == rank) continue;
            
            // 1. 发布第一段数据的非阻塞接收
            ptmp = (char*)rbuf + (ptrdiff_t)i * (ptrdiff_t)rcount * extent;
            ret = MCA_PML_CALL(irecv(ptmp, first_segment_count, rdtype, i,
                                     MCA_COLL_BASE_TAG_GATHER, comm,
                                     &first_segment_req));
            
            // 2. 发送零字节同步消息
            ret = MCA_PML_CALL(send(rbuf, 0, MPI_BYTE, i,
                                    MCA_COLL_BASE_TAG_GATHER,
                                    MCA_PML_BASE_SEND_STANDARD, comm));
            
            // 3. 发布第二段数据的非阻塞接收
            ptmp = (char*)rbuf + ((ptrdiff_t)i * (ptrdiff_t)rcount + first_segment_count) * extent;
            ret = MCA_PML_CALL(irecv(ptmp, (rcount - first_segment_count),
                                     rdtype, i, MCA_COLL_BASE_TAG_GATHER, comm,
                                     &reqs[i]));
            
            // 4. 等待第一段数据完成
            ret = ompi_request_wait(&first_segment_req, MPI_STATUS_IGNORE);
        }
        
        // 复制本地数据
        if (MPI_IN_PLACE != sbuf) {
            ret = ompi_datatype_sndrcv((void *)sbuf, scount, sdtype,
                                       (char*)rbuf + (ptrdiff_t)rank * (ptrdiff_t)rcount * extent,
                                       rcount, rdtype);
        }
        
        // 等待所有第二段数据完成
        ret = ompi_request_wait_all(size, reqs, MPI_STATUSES_IGNORE);
    }
    
    return MPI_SUCCESS;
}
```

#figure(
  code,
  caption: [线性同步Gather算法核心代码]
)\ #v(-16pt)

#let add = [
  `first_segment_size`参数的作用机制：
  1. 控制第一段数据的大小，影响同步粒度
  2. 较小的段大小提供更细粒度的同步控制
  3. 较大的段大小减少通信轮数但降低同步效果
  4. 通过`COLL_BASE_COMPUTED_SEGCOUNT`宏根据数据类型大小计算实际段数量
]

算法复杂度分析：线性同步聚集算法的时间复杂度为$O(2(p-1)α + (p-1)β m')$，其中同步机制引入额外的延迟开销。延迟复杂度为$O(p)$，包含同步消息的往返时间；带宽复杂度为$O(m'p)$，数据传输量与标准线性算法相同。该算法通过`first_segment_size`参数#footnote(add)控制同步粒度，在可靠性和性能之间提供可调节的权衡。

适用场景包括需要严格数据顺序的应用、不可靠网络环境、需要错误恢复能力的系统，以及对数据完整性要求极高的场景。该算法通过同步机制确保数据传输的有序性和可靠性，虽然增加了通信开销，但在关键应用中提供了重要的可靠性保证。

==== 线性算法（Linear Algorithm）

*函数*：`ompi_coll_base_gather_intra_basic_linear()`

源码文件路径：#link("ompi/mca/coll/base/coll_base_gather.c")

其主要原理是：所有非根进程依次向根进程发送数据，根进程循环接收来自各进程的数据，按进程号顺序存储到接收缓冲区中。

#let code = ```c
int
ompi_coll_base_gather_intra_basic_linear(const void *sbuf, size_t scount,
                                          struct ompi_datatype_t *sdtype,
                                          void *rbuf, size_t rcount,
                                          struct ompi_datatype_t *rdtype,
                                          int root,
                                          struct ompi_communicator_t *comm,
                                          mca_coll_base_module_t *module)
{
    int i, err, rank, size;
    char *ptmp;
    MPI_Aint incr, extent, lb;
    
    size = ompi_comm_size(comm);
    rank = ompi_comm_rank(comm);
    
    // 非根进程：发送数据并返回
    if (rank != root) {
        return MCA_PML_CALL(send(sbuf, scount, sdtype, root,
                                 MCA_COLL_BASE_TAG_GATHER,
                                 MCA_PML_BASE_SEND_STANDARD, comm));
    }
    
    // 根进程：循环接收数据
    ompi_datatype_get_extent(rdtype, &lb, &extent);
    incr = extent * (ptrdiff_t)rcount;
    
    for (i = 0, ptmp = (char *) rbuf; i < size; ++i, ptmp += incr) {
        if (i == rank) {
            // 处理本地数据
            if (MPI_IN_PLACE != sbuf) {
                err = ompi_datatype_sndrcv((void *)sbuf, scount, sdtype,
                                           ptmp, rcount, rdtype);
            } else {
                err = MPI_SUCCESS;
            }
        } else {
            // 从其他进程接收数据
            err = MCA_PML_CALL(recv(ptmp, rcount, rdtype, i,
                                    MCA_COLL_BASE_TAG_GATHER,
                                    comm, MPI_STATUS_IGNORE));
        }
        if (MPI_SUCCESS != err) return err;
    }
    
    return MPI_SUCCESS;
}
```

#figure(
  code,
  caption: [线性Gather算法核心代码]
)\ #v(-16pt)

算法复杂度分析：线性聚集算法的时间复杂度为$O((p-1)α + (p-1)β m')$，其中$m' = "scount"$为单个进程的数据大小。延迟复杂度为$O(p)$，根进程需要进行$p-1$次串行接收操作；带宽复杂度为$O(m'p)$，总传输量为所有进程数据之和。空间复杂度为$O(1)$。

适用场景包括小规模通信子($p≤4$)、小消息收集、网络连接性差的环境，以及作为复杂算法的回退选择。该算法的主要优势是实现简单、无拓扑构建开销，但在大规模场景下根进程会成为严重瓶颈，扩展性较差。

==== 其它Gather算法

除了上述实现的算法外，源码注释中提到了以下待实现或优化的Gather算法和优化策略：

#list(
[通用树形算法（`ompi_coll_base_gather_intra_generic`）\
提供通用的树形收集框架，可配合不同树结构（如二叉树、k进制树等）实现灵活的收集策略。],

[二进制树算法（`ompi_coll_base_gather_intra_binary`）\
使用完全二叉树结构进行数据收集，在某些场景下可能比二项式树有更好的负载平衡。],

[链式收集算法（`ompi_coll_base_gather_intra_chain`）\
形成一个或多个通信链，数据沿链向根进程汇聚，适合特定网络拓扑结构。],

[流水线收集算法（`ompi_coll_base_gather_intra_pipeline`）\
将大消息分段，采用流水线方式在树形或链式结构上收集数据，提升大消息处理效率。],

[消息分段优化\
对于超大消息，采用分段传输策略，结合流水线技术实现更好的内存利用和通信重叠。],

[基于网络拓扑的优化算法\
根据具体网络拓扑（如胖树、环面等）优化数据收集路径，减少网络拥塞并提高带宽利用率。] )\ #v(-16pt)

==== 总结

#let add = [
  源码注释中指出其中部分代码仍待开发：\

  _Todo: gather_intra_generic, gather_intra_binary, gather_intra_chain, gather_intra_pipeline, segmentation?_
]

基于上述对`MPI_Gather`的算法#footnote(add)的讨论，整理得如下表格：

#let gather_summary = table(
  columns: (0.9fr, 2fr, 0.9fr, 2fr, 1.5fr),
  align: (left, left, left, left, left),
  stroke: 0.5pt,
  table.header(
    [*算法名称*], 
    [*函数名称*], 
    [*可选参数*], 
    [*时间复杂度*], 
    [*适用场景*]
  ),
  
  [二项式树算法], 
  [#text(size: 8pt)[`ompi_coll_base_gather_`\ `intra_binomial`]], 
  [无], 
  [$O(α log(p) + β m'(p-1))$], 
  [大规模通信子\ 大消息收集],
  
  [线性同步算法], 
  [#text(size: 8pt)[`ompi_coll_base_gather_`\ `intra_linear_sync`]], 
  [`first_seg`\ `ment_size`], 
  [$O(2(p-1)α + (p-1)β m')$], 
  [可靠性要求高\ 不可靠网络环境],
  
  [线性算法], 
  [#text(size: 8pt)[`ompi_coll_base_gather_`\ `intra_basic_linear`]], 
  [无], 
  [$O((p-1)α + (p-1)β m')$], 
  [小规模通信子\ 或回退选择],
  
  [通用树形算法], 
  [#text(size: 8pt)[`ompi_coll_base_gather_`\ `intra_generic`]], 
  [#text(size: 8pt)[`tree`\ `segcount`]], 
  [取决于树结构], 
  [通用框架\ 配合不同树结构],
  
  [二进制树算法], 
  [#text(size: 8pt)[`ompi_coll_base_gather_`\ `intra_binary`]], 
  [`segsize`], 
  [$O(log_2(p)α + β m')$], 
  [完全二叉树结构\ 负载平衡优化],
  
  [链式收集算法], 
  [#text(size: 8pt)[`ompi_coll_base_gather_`\ `intra_chain`]], 
  [#text(size: 8pt)[`segsize`\ `chains`]], 
  [$O(p/"chains" dot α + β m')$], 
  [特定网络拓扑\ 多链并行收集],
  
  [流水线收集算法], 
  [#text(size: 8pt)[`ompi_coll_base_gather_`\ `intra_pipeline`]], 
  [`segsize`], 
  [$O((log_2(p) + S)α + β m')$], 
  [大消息收集\ 流水线重叠],
)

#gather_summary

#align(center)[
  #text[表 3.3：Open MPI Gather算法总结]
]

#align(left)[
  #columns(2)[
    *参数说明：*
    - $S$: 流水线算法中的段数
    - $α$: 通信延迟参数，$β$: 带宽倒数参数
    - $m'$: 单个进程数据大小（scount），$p$: 进程数量
    - `first_segment_size`: 控制同步算法第一段数据大小
    - `chains`: 链式算法中并行链的数量
    
    #colbreak()
    \
    - `tree`: 指定使用的树结构类型
    - `segcount`: 每段传输的元素数量
    - `segsize`: 控制消息分段大小的参数
  ]
]

=== Allgather

=== Allgather

`Allgather`的函数原型如下：

```c
MPI_Allgather(
    void* send_data,
    int send_count,
    MPI_Datatype send_type,
    void* recv_data,
    int recv_count,
    MPI_Datatype recv_type,
    MPI_Comm communicator)
```\ #v(-16pt)

其中：`send_data`是每个进程要发送的数据，`recv_data`是所有进程接收所有数据的缓冲区。与Gather不同，Allgather中每个进程都能获得完整的聚集结果，无需指定根进程。`send_count`和`recv_count`分别指定每个进程发送和从每个进程接收的数据元素数量。

Open MPI为Allgather操作提供了丰富的算法实现：

==== 递归加倍算法（Recursive Doubling Algorithm）

*函数*：`ompi_coll_base_allgather_intra_recursivedoubling()`

源码文件路径：#link("ompi/mca/coll/base/coll_base_allgather.c")

其主要原理是：每轮通信距离加倍，数据量加倍，通过$log_2(p)$轮交换实现全收集。相比Gather的单向收集，此算法实现双向并行数据交换。

#let code = ```c
int
ompi_coll_base_allgather_intra_recursivedoubling(const void *sbuf, size_t scount,
                                                  struct ompi_datatype_t *sdtype,
                                                  void* rbuf, size_t rcount,
                                                  struct ompi_datatype_t *rdtype,
                                                  struct ompi_communicator_t *comm,
                                                  mca_coll_base_module_t *module)
{
    // 检查是否为2的幂次进程数
    pow2size = opal_next_poweroftwo (size);
    pow2size >>=1;
    
    if (pow2size != size) {
        // 非2的幂次时回退到Bruck算法
        int k = 2;
        return ompi_coll_base_allgather_intra_k_bruck(sbuf, scount, sdtype,
                                                      rbuf, rcount, rdtype,
                                                      comm, module, k);
    }
    
    // 初始化：复制本地数据到接收缓冲区
    if (MPI_IN_PLACE != sbuf) {
        tmpsend = (char*) sbuf;
        tmprecv = (char*) rbuf + (ptrdiff_t)rank * (ptrdiff_t)rcount * rext;
        err = ompi_datatype_sndrcv(tmpsend, scount, sdtype, tmprecv, rcount, rdtype);
    }
    
    // 递归加倍通信循环
    sendblocklocation = rank;
    for (distance = 0x1; distance < size; distance<<=1) {
        remote = rank ^ distance;  // XOR操作确定通信伙伴
        
        if (rank < remote) {
            tmpsend = (char*)rbuf + (ptrdiff_t)sendblocklocation * (ptrdiff_t)rcount * rext;
            tmprecv = (char*)rbuf + (ptrdiff_t)(sendblocklocation + distance) * (ptrdiff_t)rcount * rext;
        } else {
            tmpsend = (char*)rbuf + (ptrdiff_t)sendblocklocation * (ptrdiff_t)rcount * rext;
            tmprecv = (char*)rbuf + (ptrdiff_t)(sendblocklocation - distance) * (ptrdiff_t)rcount * rext;
            sendblocklocation -= distance;
        }
        
        // 与远程进程交换数据块
        err = ompi_coll_base_sendrecv(tmpsend, (ptrdiff_t)distance * (ptrdiff_t)rcount, rdtype,
                                       remote, MCA_COLL_BASE_TAG_ALLGATHER,
                                       tmprecv, (ptrdiff_t)distance * (ptrdiff_t)rcount, rdtype,
                                       remote, MCA_COLL_BASE_TAG_ALLGATHER,
                                       comm, MPI_STATUS_IGNORE, rank);
    }
    
    return OMPI_SUCCESS;
}
```

#figure(
  code,
  caption: [递归加倍Allgather算法核心代码]
)\ #v(-16pt)

算法复杂度分析：递归加倍算法的时间复杂度为$O(α log_2(p) + β m'(p-1))$，其中$m' = "scount"$。延迟复杂度为$O(log_2 p)$，是Gather线性算法的显著改进；带宽复杂度为$O(m'p)$，接近理论最优。该算法通过XOR操作确定通信伙伴，实现完美的负载均衡，但目前限制于2的幂次进程数。

适用场景包括2的幂次规模通信子、延迟敏感应用、需要最小通信轮数的场景。相比Gather算法需要额外的广播阶段，递归加倍直接实现全收集，在支持的进程数范围内提供最优性能。

==== Sparbit算法

*函数*：`ompi_coll_base_allgather_intra_sparbit()`

源码文件路径：#link("ompi/mca/coll/base/coll_base_allgather.c")

其主要原理是：类似Bruck算法但采用反向距离和递增数据大小的对数级算法，通过稀疏位向量优化实现数据局部性感知的全收集。

#let code = ```c
int ompi_coll_base_allgather_intra_sparbit(const void *sbuf, size_t scount,
                                                  struct ompi_datatype_t *sdtype,
                                                  void* rbuf, size_t rcount,
                                                  struct ompi_datatype_t *rdtype,
                                                  struct ompi_communicator_t *comm,
                                                  mca_coll_base_module_t *module)
{
    // 初始化通信参数
    comm_size = ompi_comm_size(comm);
    rank = ompi_comm_rank(comm);
    comm_log = ceil(log(comm_size)/log(2));
    distance <<= comm_log - 1;
    
    // 计算排除步骤的位掩码
    last_ignore = __builtin_ctz(comm_size);
    ignore_steps = (~((uint32_t) comm_size >> last_ignore) | 1) << last_ignore;
    
    // 执行对数级通信循环
    for (int i = 0; i < comm_log; ++i) {
       sendto = (rank + distance) % comm_size;  
       recvfrom = (rank - distance + comm_size) % comm_size;  
       exclusion = (distance & ignore_steps) == distance;

       // 非阻塞多块数据传输
       for (transfer_count = 0; transfer_count < data_expected - exclusion; transfer_count++) {
           send_disp = (rank - 2 * transfer_count * distance + comm_size) % comm_size;
           recv_disp = (rank - (2 * transfer_count + 1) * distance + comm_size) % comm_size;

           // 使用不同标签避免消息冲突
           MCA_PML_CALL(isend(tmpsend + (ptrdiff_t) send_disp * scount * rext, scount, rdtype, 
                              sendto, MCA_COLL_BASE_TAG_HCOLL_BASE - send_disp, 
                              MCA_PML_BASE_SEND_STANDARD, comm, requests + transfer_count));
           MCA_PML_CALL(irecv(tmprecv + (ptrdiff_t) recv_disp * rcount * rext, rcount, rdtype, 
                              recvfrom, MCA_COLL_BASE_TAG_HCOLL_BASE - recv_disp, 
                              comm, requests + data_expected - exclusion + transfer_count));
       }
       ompi_request_wait_all(transfer_count * 2, requests, MPI_STATUSES_IGNORE);

       distance >>= 1; 
       data_expected = (data_expected << 1) - exclusion;
       exclusion = 0;
    }
    
    free(requests);
    return OMPI_SUCCESS;
}
```

#figure(
  code,
  caption: [Sparbit Allgather算法核心代码]
)\ #v(-16pt)

#let add = [
  源码注释中指出该算法在《Sparbit: a new logarithmic-cost and data locality-aware MPI Allgather algorithm》中详细描述
]

算法复杂度分析：Sparbit算法的时间复杂度为$O(α log(p) + β m'(p-1))$，与递归加倍相当但数据访问模式更优。该算法通过反向距离#footnote(add)和逐步增加的数据传输量实现更好的缓存局部性，在某些架构上可能优于传统算法。

适用场景包括任意进程数的通信子、对内存访问模式敏感的应用、具有复杂内存层次结构的系统。相比递归加倍的进程数限制，Sparbit提供了更通用的解决方案。

==== 环形算法（Ring Algorithm）

*函数*：`ompi_coll_base_allgather_intra_ring()`

源码文件路径：#link("ompi/mca/coll/base/coll_base_allgather.c")

其主要原理是：每个进程将数据发送给右邻居，从左邻居接收数据，通过$p-1$轮通信实现全收集。与Gather的树形收集不同，环形算法提供完美的负载均衡。

#let code = ```c
int ompi_coll_base_allgather_intra_ring(const void *sbuf, size_t scount,
                                         struct ompi_datatype_t *sdtype,
                                         void* rbuf, size_t rcount,
                                         struct ompi_datatype_t *rdtype,
                                         struct ompi_communicator_t *comm,
                                         mca_coll_base_module_t *module)
{
    // 初始化：复制本地数据
    tmprecv = (char*) rbuf + (ptrdiff_t)rank * (ptrdiff_t)rcount * rext;
    if (MPI_IN_PLACE != sbuf) {
        tmpsend = (char*) sbuf;
        err = ompi_datatype_sndrcv(tmpsend, scount, sdtype, tmprecv, rcount, rdtype);
    }
    
    // 确定环形通信的邻居
    sendto = (rank + 1) % size;
    recvfrom  = (rank - 1 + size) % size;
    
    // 执行环形数据传递
    for (i = 0; i < size - 1; i++) {
        recvdatafrom = (rank - i - 1 + size) % size;  // 接收数据的原始来源
        senddatafrom = (rank - i + size) % size;      // 发送数据的原始来源
        
        tmprecv = (char*)rbuf + (ptrdiff_t)recvdatafrom * (ptrdiff_t)rcount * rext;
        tmpsend = (char*)rbuf + (ptrdiff_t)senddatafrom * (ptrdiff_t)rcount * rext;
        
        // 同时发送和接收数据
        err = ompi_coll_base_sendrecv(tmpsend, rcount, rdtype, sendto,
                                       MCA_COLL_BASE_TAG_ALLGATHER,
                                       tmprecv, rcount, rdtype, recvfrom,
                                       MCA_COLL_BASE_TAG_ALLGATHER,
                                       comm, MPI_STATUS_IGNORE, rank);
    }
    
    return OMPI_SUCCESS;
}
```

#figure(
  code,
  caption: [环形Allgather算法核心代码]
)\ #v(-16pt)

算法复杂度分析：环形算法的时间复杂度为$O((p-1)α + (p-1)β m')$。延迟复杂度为$O(p)$，高于对数级算法但避免了根进程瓶颈；带宽复杂度为$O(m'p)$，每个数据块被传输$p-1$次，接近最优。该算法的主要优势是完美的负载均衡和对任意进程数的支持。

适用场景包括带宽受限的网络环境、需要避免热点节点的场景、不规则进程数的通信子。相比Gather需要根进程处理所有数据，环形算法将负载均匀分布到所有进程。

==== 邻居交换算法（Neighbor Exchange Algorithm）

*函数*：`ompi_coll_base_allgather_intra_neighborexchange()`

源码文件路径：#link("ompi/mca/coll/base/coll_base_allgather.c")

其主要原理是：进程与直接邻居交换数据，然后扩大交换范围，通过$p/2$步实现全收集。仅适用于偶数个进程，奇数时自动回退到环形算法。

#let code = ```c
int
ompi_coll_base_allgather_intra_neighborexchange(const void *sbuf, size_t scount,
                                                 struct ompi_datatype_t *sdtype,
                                                 void* rbuf, size_t rcount,
                                                 struct ompi_datatype_t *rdtype,
                                                 struct ompi_communicator_t *comm,
                                                 mca_coll_base_module_t *module)
{
    if (size % 2) {
        // 奇数进程数时回退到环形算法
        return ompi_coll_base_allgather_intra_ring(sbuf, scount, sdtype,
                                                    rbuf, rcount, rdtype,
                                                    comm, module);
    }
    
    // 根据奇偶性确定邻居和数据流向
    even_rank = !(rank % 2);
    if (even_rank) {
        neighbor[0] = (rank + 1) % size;
        neighbor[1] = (rank - 1 + size) % size;
        recv_data_from[0] = rank;
        recv_data_from[1] = rank;
        offset_at_step[0] = (+2);
        offset_at_step[1] = (-2);
    } else {
        neighbor[0] = (rank - 1 + size) % size;
        neighbor[1] = (rank + 1) % size;
        recv_data_from[0] = neighbor[0];
        recv_data_from[1] = neighbor[0];
        offset_at_step[0] = (-2);
        offset_at_step[1] = (+2);
    }
    
    // 第一步：与直接邻居交换单个数据块
    tmprecv = (char*)rbuf + (ptrdiff_t)neighbor[0] * (ptrdiff_t)rcount * rext;
    tmpsend = (char*)rbuf + (ptrdiff_t)rank * (ptrdiff_t)rcount * rext;
    err = ompi_coll_base_sendrecv(tmpsend, rcount, rdtype, neighbor[0],
                                   MCA_COLL_BASE_TAG_ALLGATHER,
                                   tmprecv, rcount, rdtype, neighbor[0],
                                   MCA_COLL_BASE_TAG_ALLGATHER,
                                   comm, MPI_STATUS_IGNORE, rank);
    
    // 后续步骤：交换逐步增大的数据块
    send_data_from = even_rank ? rank : recv_data_from[0];
    
    for (i = 1; i < (size / 2); i++) {
        const int i_parity = i % 2;
        recv_data_from[i_parity] =
            (recv_data_from[i_parity] + offset_at_step[i_parity] + size) % size;
        
        tmprecv = (char*)rbuf + (ptrdiff_t)recv_data_from[i_parity] * (ptrdiff_t)rcount * rext;
        tmpsend = (char*)rbuf + (ptrdiff_t)send_data_from * rcount * rext;
        
        // 交换两个数据块
        err = ompi_coll_base_sendrecv(tmpsend, (ptrdiff_t)2 * (ptrdiff_t)rcount, rdtype,
                                       neighbor[i_parity], MCA_COLL_BASE_TAG_ALLGATHER,
                                       tmprecv, (ptrdiff_t)2 * (ptrdiff_t)rcount, rdtype,
                                       neighbor[i_parity], MCA_COLL_BASE_TAG_ALLGATHER,
                                       comm, MPI_STATUS_IGNORE, rank);
        
        send_data_from = recv_data_from[i_parity];
    }
    
    return OMPI_SUCCESS;
}
```

#figure(
  code,
  caption: [邻居交换Allgather算法核心代码]
)\ #v(-16pt)

#let add = [
  邻居交换算法的特点：
  1. *进程数限制*：仅适用于偶数个进程
  2. *双向交换*：每步都进行双向同时交换，提高效率
  3. *渐进增大*：交换的数据块大小逐步增大
  4. *回退机制*：奇数进程时自动回退到环形算法
]

算法复杂度分析：邻居交换算法的时间复杂度为$O((p/2)α + (p-1)β m')$。延迟复杂度为$O(p/2)$，比环形算法减半；带宽复杂度为$O(m'p)$，通过双向同时交换#footnote(add)获得更好的带宽利用率。

适用场景包括偶数个进程的通信子、需要减少通信轮数的应用、具有良好双向带宽的网络环境。相比环形算法的单向传递，邻居交换通过双向并行传输提升效率。

==== K-Bruck算法

*函数*：`ompi_coll_base_allgather_intra_k_bruck()`

源码文件路径：#link("ompi/mca/coll/base/coll_base_allgather.c")

其主要原理是：扩展的Bruck算法，支持任意基数k，通过$log_k(p)$步实现全收集，利用非阻塞通信充分利用多端口优势。

#let code = ```c
int ompi_coll_base_allgather_intra_k_bruck(const void *sbuf, size_t scount,
                                          struct ompi_datatype_t *sdtype,
                                          void* rbuf, size_t rcount,
                                          struct ompi_datatype_t *rdtype,
                                          struct ompi_communicator_t *comm,
                                          mca_coll_base_module_t *module,
                                          int radix)
{
    // 为非0进程分配临时缓冲区用于数据重排
    if (0 != rank) {
        rsize = opal_datatype_span(&rdtype->super, (size_t)rcount * (size - rank), &rgap);
        tmp_buf = (char *) malloc(rsize);
        tmp_buf_start = tmp_buf - rgap;
    }
    
    // 执行k-进制通信循环
    max_reqs = 2 * (radix - 1);
    reqs = ompi_coll_base_comm_get_reqs(module->base_data, max_reqs);
    recvcount = 1;
    tmpsend = (char*) rbuf;
    
    for (distance = 1; distance < size; distance *= radix) {
        num_reqs = 0;
        for (int j = 1; j < radix; j++) {
            if (distance * j >= size) break;
            
            src = (rank + distance * j) % size;
            dst = (rank - distance * j + size) % size;
            tmprecv = tmpsend + (ptrdiff_t)distance * j * rcount * rextent;
            
            // 计算传输数据量
            if (distance <= (size / radix)) {
                recvcount = distance;
            } else {
                recvcount = (distance < (size - distance * j)?
                            distance:(size - distance * j));
            }
            
            // 非阻塞发送和接收
            err = MCA_PML_CALL(irecv(tmprecv, recvcount * rcount, rdtype, src,
                                     MCA_COLL_BASE_TAG_ALLGATHER, comm, &reqs[num_reqs++]));
            err = MCA_PML_CALL(isend(tmpsend, recvcount * rcount, rdtype, dst,
                                     MCA_COLL_BASE_TAG_ALLGATHER, 
                                     MCA_PML_BASE_SEND_STANDARD, comm, &reqs[num_reqs++]));
        }
        err = ompi_request_wait_all(num_reqs, reqs, MPI_STATUSES_IGNORE);
    }
    
    // 最终数据重排（除rank 0外）
    if (0 != rank) {
        // 三步数据重排序以获得正确的进程顺序
        err = ompi_datatype_copy_content_same_ddt(rdtype, 
                                                  ((ptrdiff_t)(size - rank) * rcount),
                                                  tmp_buf_start, rbuf);
        
        tmpsend = (char*) rbuf + (ptrdiff_t)(size - rank) * rcount * rextent;
        err = ompi_datatype_copy_content_same_ddt(rdtype, (ptrdiff_t)rank * rcount,
                                                  rbuf, tmpsend);
        
        tmprecv = (char*) rbuf + (ptrdiff_t)rank * rcount * rextent;
        err = ompi_datatype_copy_content_same_ddt(rdtype,
                                                  (ptrdiff_t)(size - rank) * rcount,
                                                  tmprecv, tmp_buf_start);
    }
    
    if(tmp_buf != NULL) free(tmp_buf);
    return MPI_SUCCESS;
}
```

#figure(
  code,
  caption: [K-Bruck Allgather算法核心代码]
)\ #v(-16pt)

#let add = [
  K-Bruck算法的优势：
  1. *可调基数*：通过radix参数调节延迟-带宽权衡
  2. *非阻塞通信*：充分利用网络的多端口能力
  3. *数据重排*：最终进行本地数据重排获得正确顺序
  4. *扩展Bruck*：基于经典Bruck算法的多端口扩展版本
]

算法复杂度分析：K-Bruck算法的时间复杂度为$O(α log_k(p) + β m'(p-1))$。通过调节radix参数#footnote(add)可以在延迟和带宽之间权衡：较大的k减少通信轮数但增加单轮复杂度。该算法支持任意进程数且具有良好的扩展性。

适用场景包括需要调节延迟-带宽权衡的应用、具有多端口网络的系统、中大规模任意进程数的通信子。相比递归加倍的进程数限制，K-Bruck提供了更灵活的解决方案。

==== 其它Allgather算法

除了上述核心算法外，Open MPI还实现了以下专用算法：

#list(
[两进程优化算法（`ompi_coll_base_allgather_intra_two_procs`）\
专门针对两进程情况的简单交换算法，直接进行单次数据交换。],

[基础线性算法（`ompi_coll_base_allgather_intra_basic_linear`）\
组合使用Gather和Broadcast实现Allgather，适合作为复杂算法的回退选择。],

[直接消息传递算法（`ompi_coll_base_allgather_direct_messaging`）\
每个进程直接与所有其他进程通信的贪心算法，避免根节点瓶颈但可能造成网络拥塞。] )\ #v(-16pt)

==== 总结

基于上述对`MPI_Allgather`的算法的讨论，整理得如下表格：

#let allgather_summary = table(
  columns: (0.9fr, 2.7fr, 0.9fr, 2.5fr, 1.6fr),
  align: (left, left, left, left, left),
  stroke: 0.5pt,
  table.header(
    [*算法名称*], 
    [*函数名称*], 
    [*可选参数*], 
    [*时间复杂度*], 
    [*适用场景*]
  ),
  
  [递归加倍算法], 
  [#text(size: 8pt)[`ompi_coll_base_allgather_`\ `intra_recursivedoubling`]], 
  [无], 
  [$O(α log_2(p) + β m'(p-1))$], 
  [2的幂次进程数\ 延迟敏感应用],
  
  [Sparbit算法], 
  [#text(size: 8pt)[`ompi_coll_base_allgather_`\ `intra_sparbit`]], 
  [无], 
  [$O(α log(p) + β m'(p-1))$], 
  [数据局部敏感\ 任意进程数],
  
  [环形算法], 
  [#text(size: 8pt)[`ompi_coll_base_allgather_`\ `intra_ring`]], 
  [无], 
  [$O((p-1)α + (p-1)β m')$], 
  [带宽受限网络\ 负载均衡需求],
  
  [邻居交换算法], 
  [#text(size: 8pt)[`ompi_coll_base_allgather_`\ `intra_neighborexchange`]], 
  [无], 
  [$O((p/2)α + (p-1)β m')$], 
  [偶数进程数\ 双向带宽充足],
  
  [K-Bruck算法], 
  [#text(size: 8pt)[`ompi_coll_base_allgather_`\ `intra_k_bruck`]], 
  [`radix`], 
  [$O(α log_k(p) + β m'(p-1))$], 
  [延迟-带宽权衡\ 多端口网络],
  
  [两进程算法], 
  [#text(size: 8pt)[`ompi_coll_base_allgather_`\ `intra_two_procs`]], 
  [无], 
  [$O(α + β m')$], 
  [两进程通信子\ 简单优化],
  
  [基础线性算法], 
  [#text(size: 8pt)[`ompi_coll_base_allgather_`\ `intra_basic_linear`]], 
  [无], 
  [$O((p-1)α + p β m')$], 
  [回退选择\ Gather+Bcast],
  
  [直接消息算法], 
  [#text(size: 8pt)[`ompi_coll_base_allgather_`\ `direct_messaging`]], 
  [无], 
  [$O((p-1)α + (p-1)β m')$], 
  [小消息低延迟\ 避免中转开销],
)

#allgather_summary

#align(center)[
  #text[表 3.4：Open MPI Allgather算法总结]
]

#align(left)[
  #columns(2)[
    *参数说明：*
    - $α$: 通信延迟参数，$β$: 带宽倒数参数
    - $m'$: 单个进程数据大小（scount），$p$: 进程数量
    - `radix`: K-Bruck算法的基数参数（k值）
  ]
]

=== Reduce

`Reduce`的函数原型如下：

```c
MPI_Reduce(
    void* send_data,
    void* recv_data,
    int count,
    MPI_Datatype datatype,
    MPI_Op op,
    int root,
    MPI_Comm communicator)
```\ #v(-16pt)

其中：`send_data`是每个进程要发送的数据，`recv_data`是根进程接收归约结果的缓冲区（仅在根进程有效）。`count`指定参与运算的数据元素数量，`op`指定归约操作（如`MPI_SUM`、`MPI_MAX`等）。与Gather操作不同，Reduce不仅收集数据，还对收集的数据执行指定的归约运算。

Open MPI为Reduce操作提供了多种算法实现：

==== 通用树形算法（Generic Tree Algorithm）

*函数*：`ompi_coll_base_reduce_generic()`

源码文件路径：#link("ompi/mca/coll/base/coll_base_reduce.c")

其主要原理是：提供通用的树形归约框架，支持任意树形拓扑结构和消息分段。非叶节点从子节点接收数据并执行归约操作，然后向父节点发送结果。支持流水线处理和非阻塞通信优化。

#let code = ```c
int ompi_coll_base_reduce_generic( const void* sendbuf, void* recvbuf, size_t original_count,
                                    ompi_datatype_t* datatype, ompi_op_t* op,
                                    int root, ompi_communicator_t* comm,
                                    mca_coll_base_module_t *module,
                                    ompi_coll_tree_t* tree, size_t count_by_segment,
                                    int max_outstanding_reqs )
{
    // 计算分段参数
    num_segments = (int)(((size_t)original_count + (size_t)count_by_segment - (size_t)1) / (size_t)count_by_segment);
    segment_increment = (ptrdiff_t)count_by_segment * extent;
    
    if( tree->tree_nextsize > 0 ) {
        // 非叶节点：接收子节点数据并执行归约
        
        // 分配累积缓冲区
        accumbuf = (char*)recvbuf;
        if( (NULL == accumbuf) || (root != rank) ) {
            size = opal_datatype_span(&datatype->super, original_count, &gap);
            accumbuf_free = (char*)malloc(size);
            accumbuf = accumbuf_free - gap;
        }
        
        // 处理非交换操作的特殊情况
        if (!ompi_op_is_commute(op) && MPI_IN_PLACE != sendbuf) {
            ompi_datatype_copy_content_same_ddt(datatype, original_count,
                                                (char*)accumbuf, (char*)sendtmpbuf);
        }
        
        // 分段流水线处理
        for( segindex = 0; segindex <= num_segments; segindex++ ) {
            for( i = 0; i < tree->tree_nextsize; i++ ) {
                // 发布非阻塞接收
                if( segindex < num_segments ) {
                    ret = MCA_PML_CALL(irecv(local_recvbuf, recvcount, datatype,
                                             tree->tree_next[i], MCA_COLL_BASE_TAG_REDUCE,
                                             comm, &reqs[inbi]));
                }
                
                // 等待前一个请求完成并执行归约
                ret = ompi_request_wait(&reqs[inbi ^ 1], MPI_STATUSES_IGNORE);
                local_op_buffer = inbuf[inbi ^ 1];
                
                // 执行归约操作
                if( i > 0 ) {
                    ompi_op_reduce(op, local_op_buffer,
                                   accumbuf + (ptrdiff_t)segindex * (ptrdiff_t)segment_increment,
                                   recvcount, datatype );
                }
                
                // 向父节点发送累积结果
                if (rank != tree->tree_root && segindex > 0) {
                    ret = MCA_PML_CALL( send( accumulator, prevcount, datatype, tree->tree_prev,
                                              MCA_COLL_BASE_TAG_REDUCE, MCA_PML_BASE_SEND_STANDARD, comm) );
                }
                
                inbi = inbi ^ 1;
            }
        }
    } else {
        // 叶节点：发送数据到父节点
        
        if ((0 == max_outstanding_reqs) || (num_segments <= max_outstanding_reqs)) {
            // 使用阻塞发送
            segindex = 0;
            while ( original_count > 0) {
                ret = MCA_PML_CALL( send((char*)sendbuf + (ptrdiff_t)segindex * (ptrdiff_t)segment_increment,
                                         count_by_segment, datatype, tree->tree_prev,
                                         MCA_COLL_BASE_TAG_REDUCE, MCA_PML_BASE_SEND_STANDARD, comm) );
                segindex++;
                original_count -= count_by_segment;
            }
        } else {
            // 使用流控制的非阻塞发送
            sreq = ompi_coll_base_comm_get_reqs(module->base_data, max_outstanding_reqs);
            
            // 发送前max_outstanding_reqs个分段
            for (segindex = 0; segindex < max_outstanding_reqs; segindex++) {
                ret = MCA_PML_CALL( isend((char*)sendbuf + (ptrdiff_t)segindex * (ptrdiff_t)segment_increment,
                                          count_by_segment, datatype, tree->tree_prev,
                                          MCA_COLL_BASE_TAG_REDUCE, MCA_PML_BASE_SEND_SYNCHRONOUS,
                                          comm, &sreq[segindex]) );
                original_count -= count_by_segment;
            }
            
            // 流水线处理剩余分段
            creq = 0;
            while ( original_count > 0 ) {
                ret = ompi_request_wait(&sreq[creq], MPI_STATUS_IGNORE);
                ret = MCA_PML_CALL( isend((char*)sendbuf + (ptrdiff_t)segindex * (ptrdiff_t)segment_increment,
                                          count_by_segment, datatype, tree->tree_prev,
                                          MCA_COLL_BASE_TAG_REDUCE, MCA_PML_BASE_SEND_SYNCHRONOUS,
                                          comm, &sreq[creq]) );
                creq = (creq + 1) % max_outstanding_reqs;
                segindex++;
                original_count -= count_by_segment;
            }
            
            ret = ompi_request_wait_all( max_outstanding_reqs, sreq, MPI_STATUSES_IGNORE );
        }
    }
    
    return OMPI_SUCCESS;
}
```

#figure(
  code,
  caption: [通用树形Reduce算法核心代码]
)\ #v(-16pt)

算法复杂度分析：通用树形归约算法的时间复杂度取决于具体的树结构，通常为$O(α log(p) + β m)$，其中$m$为总数据量。该算法通过分段处理支持大消息归约，通过流水线技术实现通信-计算重叠。非交换操作需要特殊处理以保证运算顺序的正确性。

适用场景包括需要自定义树形拓扑的应用、大消息归约、需要流水线优化的高性能计算场景。该算法作为Open MPI中其他具体树形算法的基础框架，提供了灵活的参数配置和优化选项。

==== 二项式树算法（Binomial Tree Algorithm）

*函数*：`ompi_coll_base_reduce_intra_binomial()`

源码文件路径：#link("ompi/mca/coll/base/coll_base_reduce.c")

其主要原理是：使用二项式树结构进行归约，通过调用通用树形算法实现。相比Gather的单纯数据收集，该算法在每个节点执行归约运算，减少了网络传输的数据量。

#let code = ```c
int ompi_coll_base_reduce_intra_binomial( const void *sendbuf, void *recvbuf,
                                           size_t count, ompi_datatype_t* datatype,
                                           ompi_op_t* op, int root,
                                           ompi_communicator_t* comm,
                                           mca_coll_base_module_t *module,
                                           uint32_t segsize, int max_outstanding_reqs  )
{
    size_t segcount = count;
    size_t typelng;
    mca_coll_base_module_t *base_module = (mca_coll_base_module_t*) module;
    mca_coll_base_comm_t *data = base_module->base_data;

    COLL_BASE_UPDATE_IN_ORDER_BMTREE( comm, base_module, root );

    // 计算分段参数
    ompi_datatype_type_size( datatype, &typelng );
    COLL_BASE_COMPUTED_SEGCOUNT( segsize, typelng, segcount );

    // 调用通用树形算法
    return ompi_coll_base_reduce_generic( sendbuf, recvbuf, count, datatype,
                                           op, root, comm, module,
                                           data->cached_in_order_bmtree,
                                           segcount, max_outstanding_reqs );
}
```

#figure(
  code,
  caption: [二项式树Reduce算法核心代码]
)\ #v(-16pt)

算法复杂度分析：二项式树归约算法的时间复杂度为$O(α log(p) + β m)$。相比二项式树Gather的$O(α log(p) + β m(p-1))$，归约操作通过在中间节点执行运算显著减少了数据传输量。延迟复杂度为$O(log p)$，适合大规模通信子。

适用场景包括大规模通信子、大消息归约、延迟敏感的归约操作。该算法通过二项式树的平衡结构和归约运算的数据压缩特性，在大多数场景下提供优秀的性能。

==== K项树算法（K-nomial Tree Algorithm）

*函数*：`ompi_coll_base_reduce_intra_knomial()`

源码文件路径：#link("ompi/mca/coll/base/coll_base_reduce.c")

其主要原理是：使用可配置基数的k项树结构进行归约，每个节点可以有多个子节点。通过调节radix参数在延迟和并发度之间权衡，支持非阻塞接收优化。

#let code = ```c
int ompi_coll_base_reduce_intra_knomial( const void *sendbuf, void *recvbuf,
                                           size_t count, ompi_datatype_t* datatype,
                                           ompi_op_t* op, int root,
                                           ompi_communicator_t* comm,
                                           mca_coll_base_module_t *module,
                                           uint32_t segsize, int max_outstanding_reqs, int radix)
{
    // 创建k项树
    COLL_BASE_UPDATE_KMTREE(comm, base_module, root, radix);
    tree = data->cached_kmtree;
    num_children = tree->tree_nextsize;
    
    // 分配子节点数据缓冲区
    if(!is_leaf) {
        buf_size = opal_datatype_span(&datatype->super, (int64_t)count * num_children, &gap);
        child_buf = (char *)malloc(buf_size);
        child_buf_start = child_buf - gap;
        reqs = ompi_coll_base_comm_get_reqs(data, max_reqs);
    }
    
    // 非阻塞接收所有子节点数据
    for (int i = 0; i < num_children; i++) {
        int child = tree->tree_next[i];
        err = MCA_PML_CALL(irecv(child_buf_start + (ptrdiff_t)i * count * extent,
                                 count, datatype, child, MCA_COLL_BASE_TAG_REDUCE,
                                 comm, &reqs[num_reqs++]));
    }
    
    // 等待所有接收完成
    if (num_reqs > 0) {
        err = ompi_request_wait_all(num_reqs, reqs, MPI_STATUS_IGNORE);
    }
    
    // 执行归约操作
    for (int i = 0; i < num_children; i++) {
        ompi_op_reduce(op, child_buf_start + (ptrdiff_t)i * count * extent,
                       reduce_buf, count, datatype);
    }
    
    // 向父节点发送结果
    if (rank != root) {
        err = MCA_PML_CALL(send(reduce_buf_start, count, datatype, tree->tree_prev,
                                MCA_COLL_BASE_TAG_REDUCE, MCA_PML_BASE_SEND_STANDARD, comm));
    }
    
    // 根节点复制最终结果
    if (rank == root) {
        err = ompi_datatype_copy_content_same_ddt(datatype, count,
                                                  (char*)recvbuf, (char*)reduce_buf_start);
    }
    
    return MPI_SUCCESS;
}
```

#figure(
  code,
  caption: [K项树Reduce算法核心代码]
)\ #v(-16pt)

算法复杂度分析：K项树归约算法的时间复杂度为$O(α log_k(p) + β m)$。通过调节radix参数可以在通信轮数和单轮并发度之间权衡：较大的k减少延迟但增加单轮复杂度。该算法支持任意进程数且具有良好的扩展性。

适用场景包括需要调节延迟-并发度权衡的应用、具有多端口网络的系统、中大规模任意进程数的通信子。相比二项式树的固定结构，K项树提供了更灵活的性能调优选项。

==== 有序二叉树算法（In-order Binary Tree Algorithm）

*函数*：`ompi_coll_base_reduce_intra_in_order_binary()`

源码文件路径：#link("ompi/mca/coll/base/coll_base_reduce.c")

其主要原理是：专门为非交换归约操作设计的算法，使用有序二叉树确保运算顺序的正确性。必须使用进程号(size-1)作为内部根节点，最后将结果传输给实际根进程。

#let code = ```c
int ompi_coll_base_reduce_intra_in_order_binary( const void *sendbuf, void *recvbuf,
                                                  size_t count, ompi_datatype_t* datatype,
                                                  ompi_op_t* op, int root,
                                                  ompi_communicator_t* comm,
                                                  mca_coll_base_module_t *module,
                                                  uint32_t segsize, int max_outstanding_reqs  )
{
    // 有序二叉树必须使用(size-1)作为内部根节点以保证运算顺序
    io_root = size - 1;
    use_this_sendbuf = (void *)sendbuf;
    use_this_recvbuf = recvbuf;
    
    if (io_root != root) {
        dsize = opal_datatype_span(&datatype->super, count, &gap);
        
        if ((root == rank) && (MPI_IN_PLACE == sendbuf)) {
            // 实际根进程使用IN_PLACE时的特殊处理
            tmpbuf_free = (char *) malloc(dsize);
            tmpbuf = tmpbuf_free - gap;
            ompi_datatype_copy_content_same_ddt(datatype, count,
                                                (char*)tmpbuf, (char*)recvbuf);
            use_this_sendbuf = tmpbuf;
        } else if (io_root == rank) {
            // 内部根进程分配临时接收缓冲区
            tmpbuf_free = (char *) malloc(dsize);
            tmpbuf = tmpbuf_free - gap;
            use_this_recvbuf = tmpbuf;
        }
    }
    
    // 使用有序二叉树执行归约
    ret = ompi_coll_base_reduce_generic( use_this_sendbuf, use_this_recvbuf, count, datatype,
                                          op, io_root, comm, module,
                                          data->cached_in_order_bintree,
                                          segcount, max_outstanding_reqs );
    
    // 处理内部根与实际根不同的情况
    if (io_root != root) {
        if (root == rank) {
            // 实际根进程从内部根接收最终结果
            ret = MCA_PML_CALL(recv(recvbuf, count, datatype, io_root,
                                    MCA_COLL_BASE_TAG_REDUCE, comm, MPI_STATUS_IGNORE));
        } else if (io_root == rank) {
            // 内部根进程向实际根发送最终结果
            ret = MCA_PML_CALL(send(use_this_recvbuf, count, datatype, root,
                                    MCA_COLL_BASE_TAG_REDUCE, MCA_PML_BASE_SEND_STANDARD, comm));
        }
    }
    
    return MPI_SUCCESS;
}
```

#figure(
  code,
  caption: [有序二叉树Reduce算法核心代码]
)\ #v(-16pt)

算法复杂度分析：有序二叉树归约算法的时间复杂度为$O(α log(p) + β m)$，与标准二叉树相同，但增加了根节点间数据传输的开销。该算法确保了非交换操作的运算顺序正确性，这是处理诸如矩阵乘法、字符串连接等非交换操作的关键。

适用场景包括非交换归约操作、需要严格运算顺序的数值计算、字符串处理等应用。该算法是Open MPI中专门处理非交换操作的重要实现，确保了数学运算的正确性。

==== 分散-聚集算法（Reduce-scatter-gather Algorithm）

*函数*：`ompi_coll_base_reduce_intra_redscat_gather()`

源码文件路径：#link("ompi/mca/coll/base/coll_base_reduce.c")

其主要原理是：实现Rabenseifner算法，先执行reduce-scatter将数据分散到各进程并执行部分归约，再通过二项式树gather收集最终结果。适合大规模归约操作，特别是当数据量大于进程数时。

#let code = ```c
int ompi_coll_base_reduce_intra_redscat_gather(
    const void *sbuf, void *rbuf, size_t count, struct ompi_datatype_t *dtype,
    struct ompi_op_t *op, int root, struct ompi_communicator_t *comm,
    mca_coll_base_module_t *module)
{
    // 第一步：处理非2的幂次进程数
    int nprocs_rem = comm_size - nprocs_pof2;
    
    if (rank < 2 * nprocs_rem) {
        int count_lhalf = count / 2;
        int count_rhalf = count - count_lhalf;
        
        if (rank % 2 != 0) {
            // 奇数进程：与左邻居交换并归约右半部分
            err = ompi_coll_base_sendrecv(rbuf, count_lhalf, dtype, rank - 1,
                                          MCA_COLL_BASE_TAG_REDUCE,
                                          (char *)tmp_buf + (ptrdiff_t)count_lhalf * extent,
                                          count_rhalf, dtype, rank - 1,
                                          MCA_COLL_BASE_TAG_REDUCE, comm, MPI_STATUS_IGNORE, rank);
            
            ompi_op_reduce(op, (char *)tmp_buf + (ptrdiff_t)count_lhalf * extent,
                           (char *)rbuf + count_lhalf * extent, count_rhalf, dtype);
            
            err = MCA_PML_CALL(send((char *)rbuf + (ptrdiff_t)count_lhalf * extent,
                                    count_rhalf, dtype, rank - 1,
                                    MCA_COLL_BASE_TAG_REDUCE, MCA_PML_BASE_SEND_STANDARD, comm));
            vrank = -1;  // 不参与后续阶段
        } else {
            // 偶数进程：与右邻居交换并归约左半部分
            err = ompi_coll_base_sendrecv((char *)rbuf + (ptrdiff_t)count_lhalf * extent,
                                          count_rhalf, dtype, rank + 1, MCA_COLL_BASE_TAG_REDUCE,
                                          tmp_buf, count_lhalf, dtype, rank + 1,
                                          MCA_COLL_BASE_TAG_REDUCE, comm, MPI_STATUS_IGNORE, rank);
            
            ompi_op_reduce(op, tmp_buf, rbuf, count_lhalf, dtype);
            
            err = MCA_PML_CALL(recv((char *)rbuf + (ptrdiff_t)count_lhalf * extent,
                                    count_rhalf, dtype, rank + 1,
                                    MCA_COLL_BASE_TAG_REDUCE, comm, MPI_STATUS_IGNORE));
            vrank = rank / 2;
        }
    } else {
        vrank = rank - nprocs_rem;
    }
    
    // 第二步：递归减半的reduce-scatter
    if (vrank != -1) {
        step = 0;
        wsize = count;
        
        for (int mask = 1; mask < nprocs_pof2; mask <<= 1) {
            int vdest = vrank ^ mask;
            int dest = (vdest < nprocs_rem) ? vdest * 2 : vdest + nprocs_rem;
            
            // 确定发送和接收的数据范围
            if (rank < dest) {
                rcount[step] = wsize / 2;
                scount[step] = wsize - rcount[step];
                sindex[step] = rindex[step] + rcount[step];
            } else {
                scount[step] = wsize / 2;
                rcount[step] = wsize - scount[step];
                rindex[step] = sindex[step] + scount[step];
            }
            
            // 交换数据并执行归约
            err = ompi_coll_base_sendrecv((char *)rbuf + (ptrdiff_t)sindex[step] * extent,
                                          scount[step], dtype, dest, MCA_COLL_BASE_TAG_REDUCE,
                                          (char *)tmp_buf + (ptrdiff_t)rindex[step] * extent,
                                          rcount[step], dtype, dest, MCA_COLL_BASE_TAG_REDUCE,
                                          comm, MPI_STATUS_IGNORE, rank);
            
            ompi_op_reduce(op, (char *)tmp_buf + (ptrdiff_t)rindex[step] * extent,
                           (char *)rbuf + (ptrdiff_t)rindex[step] * extent,
                           rcount[step], dtype);
            
            // 更新下一轮的窗口
            if (step + 1 < nsteps) {
                rindex[step + 1] = rindex[step];
                sindex[step + 1] = rindex[step];
                wsize = rcount[step];
                step++;
            }
        }
    }
    
    // 第三步：二项式树gather收集最终结果
    if (vrank != -1) {
        step = nsteps - 1;
        
        for (int mask = nprocs_pof2 >> 1; mask > 0; mask >>= 1) {
            int vdest = vrank ^ mask;
            int dest = (vdest < nprocs_rem) ? vdest * 2 : vdest + nprocs_rem;
            
            // 确定是发送还是接收
            vdest_tree = vdest >> step;
            vdest_tree <<= step;
            vroot_tree = vroot >> step;
            vroot_tree <<= step;
            
            if (vdest_tree == vroot_tree) {
                err = MCA_PML_CALL(send((char *)rbuf + (ptrdiff_t)rindex[step] * extent,
                                        rcount[step], dtype, dest, MCA_COLL_BASE_TAG_REDUCE,
                                        MCA_PML_BASE_SEND_STANDARD, comm));
                break;
            } else {
                err = MCA_PML_CALL(recv((char *)rbuf + (ptrdiff_t)sindex[step] * extent,
                                        scount[step], dtype, dest, MCA_COLL_BASE_TAG_REDUCE,
                                        comm, MPI_STATUS_IGNORE));
            }
            step--;
        }
    }
    
    return err;
}
```

#figure(
  code,
  caption: [分散-聚集Reduce算法核心代码]
)\ #v(-16pt)

算法复杂度分析：分散-聚集归约算法的时间复杂度为$O(α log(p) + β m)$，但具有更好的可扩展性。该算法特别适合count >= p的场景，通过reduce-scatter阶段的并行处理和gather阶段的结果收集，在大规模系统上表现优异。算法要求操作必须是交换的。

适用场景包括大规模并行系统、大数据量归约操作、高带宽网络环境。该算法是基于Rabenseifner论文的经典实现，在HPC领域广泛应用于大规模数值计算。

==== 线性算法（Linear Algorithm）

*函数*：`ompi_coll_base_reduce_intra_basic_linear()`

源码文件路径：#link("ompi/mca/coll/base/coll_base_reduce.c")

其主要原理是：所有非根进程将数据发送给根进程，根进程按相反顺序接收数据并执行归约操作。实现简单但根进程会成为瓶颈。

#let code = ```c
int ompi_coll_base_reduce_intra_basic_linear(const void *sbuf, void *rbuf, size_t count,
                                             struct ompi_datatype_t *dtype, struct ompi_op_t *op,
                                             int root, struct ompi_communicator_t *comm,
                                             mca_coll_base_module_t *module)
{
    // 非根进程：发送数据并返回
    if (rank != root) {
        err = MCA_PML_CALL(send(sbuf, count, dtype, root,
                                MCA_COLL_BASE_TAG_REDUCE, MCA_PML_BASE_SEND_STANDARD, comm));
        return err;
    }
    
    // 根进程：处理MPI_IN_PLACE情况
    if (MPI_IN_PLACE == sbuf) {
        sbuf = rbuf;
        inplace_temp_free = (char*)malloc(dsize);
        rbuf = inplace_temp_free - gap;
    }
    
    // 初始化接收缓冲区：从最高进程号开始
    if (rank == (size - 1)) {
        err = ompi_datatype_copy_content_same_ddt(dtype, count, (char*)rbuf, (char*)sbuf);
    } else {
        err = MCA_PML_CALL(recv(rbuf, count, dtype, size - 1,
                                MCA_COLL_BASE_TAG_REDUCE, comm, MPI_STATUS_IGNORE));
    }
    
    // 按降序接收数据并执行归约
    for (i = size - 2; i >= 0; --i) {
        if (rank == i) {
            inbuf = (char*)sbuf;
        } else {
            err = MCA_PML_CALL(recv(pml_buffer, count, dtype, i,
                                    MCA_COLL_BASE_TAG_REDUCE, comm, MPI_STATUS_IGNORE));
            inbuf = pml_buffer;
        }
        
        // 执行归约操作
        ompi_op_reduce(op, inbuf, rbuf, count, dtype);
    }
    
    // 处理MPI_IN_PLACE的最终复制
    if (NULL != inplace_temp_free) {
        err = ompi_datatype_copy_content_same_ddt(dtype, count, (char*)sbuf, rbuf);
        free(inplace_temp_free);
    }
    
    return MPI_SUCCESS;
}
```

#figure(
  code,
  caption: [线性Reduce算法核心代码]
)\ #v(-16pt)

算法复杂度分析：线性归约算法的时间复杂度为$O((p-1)α + (p-1)β m)$。延迟复杂度为$O(p)$，带宽复杂度为$O(p m)$，根进程成为明显瓶颈。该算法通过按降序接收数据确保了非交换操作的正确性。

适用场景包括小规模通信子、小消息归约、作为复杂算法的回退选择，以及调试和验证目的。虽然性能较差，但实现简单可靠，在某些特定场景下仍有价值。

==== 其它Reduce算法

除了上述核心算法外，Open MPI还实现了以下专用算法：

#list(
[链式归约算法（`ompi_coll_base_reduce_intra_chain`）\
形成一个或多个通信链，数据沿链向根进程归约，支持通过fanout参数控制并行链数。],

[流水线归约算法（`ompi_coll_base_reduce_intra_pipeline`）\
将大消息分段，采用流水线方式在链式结构上进行归约，提升大消息处理效率。],

[二进制树归约算法（`ompi_coll_base_reduce_intra_binary`）\
使用完全二叉树结构进行归约，通过调用通用算法框架实现。] )\ #v(-16pt)

==== 总结

基于上述对`MPI_Reduce`的算法的讨论，整理得如下表格：

#let reduce_summary = table(
  columns: (0.9fr, 2fr, 0.9fr, 2fr, 1.5fr),
  align: (left, left, left, left, left),
  stroke: 0.5pt,
  table.header(
    [*算法名称*], 
    [*函数名称*], 
    [*可选参数*], 
    [*时间复杂度*], 
    [*适用场景*]
  ),
  
  [通用树形算法], 
  [#text(size: 8pt)[`ompi_coll_base_reduce_`\ `generic`]], 
  [#text(size: 8pt)[`tree`\ `segcount`\ `max_reqs`]], 
  [取决于树结构], 
  [通用框架\ 任意树形拓扑],
  
  [二项式树算法], 
  [#text(size: 8pt)[`ompi_coll_base_reduce_`\ `intra_binomial`]], 
  [#text(size: 8pt)[`segsize`\ `max_reqs`]], 
  [$O(α log(p) + β m)$], 
  [大规模通信子\ 平衡性能需求],
  
  [K项树算法], 
  [#text(size: 8pt)[`ompi_coll_base_reduce_`\ `intra_knomial`]], 
  [#text(size: 8pt)[`segsize`\ `max_reqs`\ `radix`]], 
  [$O(α log_k(p) + β m)$], 
  [延迟-并发权衡\ 多端口网络],
  
  [有序二叉树算法], 
  [#text(size: 8pt)[`ompi_coll_base_reduce_`\ `intra_in_order_binary`]], 
  [#text(size: 8pt)[`segsize`\ `max_reqs`]], 
  [$O(α log(p) + β m)$], 
  [非交换操作\ 严格运算顺序],
  
  [分散-聚集算法], 
  [#text(size: 8pt)[`ompi_coll_base_reduce_`\ `intra_redscat_gather`]], 
  [无], 
  [$O(α log(p) + β m)$], 
  [大规模系统\ 大数据量归约],
  
  [线性算法], 
  [#text(size: 8pt)[`ompi_coll_base_reduce_`\ `intra_basic_linear`]], 
  [无], 
  [$O((p-1)α + (p-1)β m)$], 
  [小规模通信子\ 回退选择],
  
  [链式归约算法], 
  [#text(size: 8pt)[`ompi_coll_base_reduce_`\ `intra_chain`]], 
  [#text(size: 8pt)[`segsize`\ `fanout`\ `max_reqs`]], 
  [$O(p/"fanout" dot α + β m)$], 
  [特定网络拓扑\ 多链并行],
  
  [流水线算法], 
  [#text(size: 8pt)[`ompi_coll_base_reduce_`\ `intra_pipeline`]], 
  [#text(size: 8pt)[`segsize`\ `max_reqs`]], 
  [$O(α log(p) + β m)$], 
  [大消息归约\ 流水线重叠],
  
  [二进制树算法], 
  [#text(size: 8pt)[`ompi_coll_base_reduce_`\ `intra_binary`]], 
  [#text(size: 8pt)[`segsize`\ `max_reqs`]], 
  [$O(α log(p) + β m)$], 
  [完全二叉树结构\ 负载平衡],
)

#reduce_summary

#align(center)[
  #text[表 3.4：Open MPI Reduce算法总结]
]

#align(left)[
  #columns(2)[
    *参数说明：*
    - $α$: 通信延迟参数，$β$: 带宽倒数参数
    - $m$: 消息大小，$p$: 进程数量
    - `segsize`: 控制消息分段大小的参数
    - `max_reqs`: 最大未完成请求数，用于流控制
    - `radix`: K项树的基数参数（k值）
    
    #colbreak()
    \
    - `fanout`: 链式算法中的扇出参数
    - `tree`: 指定使用的树结构类型
    - `segcount`: 每段传输的元素数量
  ]
]

== 小结

本章通过深入分析Open MPI源码，系统阐述了集合通信算法的架构设计与核心实现。研究发现，Open MPI采用模块化组件架构（MCA），将集合通信实现分为`base`、`basic`、`tuned`等专门化组件，通过`mca_coll_base_comm_select()`机制根据运行时参数动态选择最优算法实现，为用户提供透明而高效的性能优化。

在算法实现层面，本章详细分析了五种核心集合通信操作的多种算法变体。Broadcast操作提供了10种算法实现，从线性算法的$O(p α + p β m)$复杂度到K项树算法的$O(log_k(p)α + β m)$最优带宽效率，涵盖了从小规模到大规模通信子的各种应用场景。Scatter操作实现了7种算法，其中二项式树算法通过$O(α log(p) + β m(p-1)/p)$的复杂度在大规模场景下有效分担根进程负载，而非阻塞线性算法通过通信重叠技术提升了中等规模应用的性能。Gather操作包含已实现的3种核心算法和4种规划中的算法，线性同步算法通过两阶段数据传输确保了不可靠网络环境下的数据完整性，体现了可靠性与性能的权衡设计。

Allgather操作提供了最丰富的8种算法实现，从递归加倍的$O(α log_2(p) + β m'(p-1))$延迟优化到环形算法的完美负载均衡，满足了无根进程全收集的多样化性能需求。Reduce操作通过10种算法实现了数据收集与运算融合的优化，通用树形算法提供了灵活的框架支持，有序二叉树算法专门处理非交换操作的运算顺序正确性，分散-聚集算法在大规模系统中展现出优异的扩展性。

在性能优化策略方面，分析发现Open MPI在算法设计中普遍采用了延迟-带宽权衡机制，通过radix、segsize等参数实现算法的性能调优；运用非阻塞通信和流水线处理技术提升带宽利用率；通过树形结构避免根进程瓶颈，环形算法实现完美负载分布；并关注数据局部性优化，如Sparbit算法等新兴实现注重缓存友好的数据访问模式。这些优化策略的综合运用使得Open MPI能够在不同网络环境和应用场景下提供高效的集合通信服务。

通过源码分析，本章系统梳理了集合通信算法的复杂度特征、适用场景和参数影响，为后续的性能建模和算法选择优化提供了完整的理论基础。同时通过阅读源码深化了对Open MPI集合通信实现的理解。