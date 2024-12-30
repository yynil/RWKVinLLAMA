# 训练流程图

```mermaid
flowchart TB
    Start([开始]) --> ParseArgs[解析命令行参数]
    ParseArgs --> InitDistributed[初始化分布式训练环境]
    InitDistributed --> LoadConfig[加载配置文件]
    
    subgraph ModelInit[模型初始化]
        direction TB
        LoadLLM[加载LLaMA模型] --> InitHybrid[创建HybridModel]
        InitHybrid --> |封装| WrapModel[封装LLaMA模型并添加RWKV特性]
        WrapModel --> |Stage 1| Stage1[仅修改注意力机制]
        WrapModel --> |Stage 2| Stage2[添加知识蒸馏功能]
        WrapModel --> |Stage 3| Stage3[准备SFT训练]
    end
    
    LoadConfig --> ModelInit
    
    ModelInit --> SetModelParams[设置模型参数训练状态]
    SetModelParams --> |Stage 1| SetStage1[仅训练self_attn参数]
    SetModelParams --> |Stage 2/3| SetStage23[所有参数可训练]
    
    SetStage1 & SetStage23 --> PrepareData[准备数据加载器]
    PrepareData --> InitDeepSpeed[初始化DeepSpeed]
    
    InitDeepSpeed --> |Stage 1| InitTeacherAttn[初始化教师注意力模块]
    InitDeepSpeed --> |Stage 2| InitTeacherModel[初始化完整教师模型]
    InitDeepSpeed --> |Stage 3| SkipTeacher[跳过教师模型初始化]
    
    InitTeacherAttn & InitTeacherModel & SkipTeacher --> InitWandb[初始化Wandb]
    
    InitWandb --> TrainingLoop[进入训练循环]
    
    subgraph TrainingLoop[训练循环]
        direction TB
        EpochStart[开始新epoch] --> BatchStart[批次开始]
        BatchStart --> UpdateLR[更新学习率]
        UpdateLR --> Forward[前向传播]
        Forward --> Backward[反向传播]
        Backward --> CheckAccum{是否达到梯度累积步数?}
        CheckAccum --> |是| OptimStep[优化器步进]
        CheckAccum --> |否| NextBatch[下一批次]
        OptimStep --> NextBatch
        NextBatch --> CheckEnd{是否结束训练?}
        CheckEnd --> |否| BatchStart
        CheckEnd --> |是| SaveCheckpoint[保存检查点]
    end
    
    SaveCheckpoint --> End([结束])
    
    classDef process fill:#f9f,stroke:#333,stroke-width:2px;
    classDef condition fill:#bbf,stroke:#333,stroke-width:2px;
    classDef modelInit fill:#fdf,stroke:#333,stroke-width:2px;
    class CheckAccum,CheckEnd condition;
    class TrainingLoop process;
    class ModelInit modelInit;
```

# Model Illustration

## Stage 1 - TimeMixer replacing Self-Attention

Original Decoder Layer:

```mermaid
flowchart TD
    subgraph DecoderLayer
        A["self_attn"] 
        A-->C["residual"]
        C-->D["post layer norm"]
        D-->B["mlp"]
    end
```

Replace the Attention to an AttentionWrapper which includes the original self_attn and a TimeMixer, the TimeMixer will learn to close the gap between the output of self_attn and the output of TimeMixer. The final output consists of the hidden states of original self_attn and the difference between the output of self_attn and TimeMixer.  Model will optimize the TimeMixer to minimize the difference between the output of self_attn and TimeMixer.:

```mermaid
flowchart TD
    Input["Input
    hidden_states,
    *args,
    **kwargs "]
    
    subgraph AttentionWrapper
    direction TB
        A["Self Attention"] 
        
        B["TimeMixer"]
        
        
        D1("Output of Self Attention")
        
        D2("Output of TimeMixer")

        E["Calculate hidden states difference
        Between Self Attention and TimeMixer"]

        A --> D1
        B --> D2
        D1 --> E
        D2 --> E
    end
    subgraph OutputOfAttentionWrapper
    direction TB
        D3("Hidden States")
        D4("Hidden States Difference")
    end

    subgraph DecoderLayerOutput
    direction TB
        Output("Hidden States")
        AttentionScore("Attention Score")
    end
    D3 --"add residulal"--> Residual("Residual")
    Residual --> PostLayerNorm("Post Layer Norm")
    PostLayerNorm --> MLP("MLP")
    MLP --> Output

    Input --> AttentionWrapper
    D1 --"select hidden state to output as output[0]"--> D3("Hidden States")
    E --"select the difference score as output[1]"--> D4("Hidden States Difference as the second output of Decoder Layer")
    D4 --> AttentionScore

    style Input fill:#f9f,stroke:#333,stroke-width:4px
    style D3 fill:#bbf,stroke:#333,stroke-width:4px
    style D1 fill:#bbf,stroke:#333,stroke-width:4px
    style D2 fill:#bbf,stroke:#333,stroke-width:4px
    style D4 fill:#bbf,stroke:#333,stroke-width:4px
    style Output fill:#bbf,stroke:#333,stroke-width:4px
    style AttentionScore fill:#bbf,stroke:#333,stroke-width:4px
```
