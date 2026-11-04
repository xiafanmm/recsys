# 推荐系统近期发展与有影响力的算法

> 调研时间：2026-11。基于公开论文、工业技术报告与会议论文整理。
> 文中线上增益（CTR、时长等）均为论文/报告自报数字，统计口径与业务场景各不相同，落地前需要自行验证。

## 0. 三分钟结论

过去十年，推荐系统经历了三次比较清晰的范式转移：

1. **特征交叉 + 深度模型（2016–2021）**：Wide&Deep、DeepFM、DIN、SIM 等把人工特征工程换成可学习的网络，"大 embedding + MLP" 成为工业标配。这一代模型今天仍是绝大多数公司的线上基线。
2. **行为序列建模（2018–2022）**：SASRec、BERT4Rec 把用户行为当成序列，Transformer 正式进入推荐；SIM 解决了超长行为序列在工业场景的落地问题。
3. **生成式推荐（2023–至今）**：TIGER 用「语义 ID」把"预测下一个物品"改写成"自回归生成下一段 ID"；Meta 的 HSTU 第一次在工业规模上证明推荐模型也遵循 scaling law；快手 OneRec 把召回和排序合并成一个端到端生成模型并上线主场景。2026 年的主要增量，是把大语言模型的语义知识与推理能力接入这条生成式链路。

如果只记五个名字：

| 算法 | 一句话定位 |
| --- | --- |
| **TIGER**（Google，2023） | 语义 ID + 生成式检索，生成式推荐的范式基石 |
| **HSTU**（Meta，2024） | 万亿参数、工业上线，证明推荐的 scaling law |
| **OneRec**（快手，2025） | 首个端到端统一召回与排序并大规模上线的生成式推荐系统 |
| **RankMixer**（字节，2025） | 排序模型的可扩展架构，参数量 ×100 而延迟基本不变 |
| **DIN / SIM**（阿里，2018/2020） | 用户行为建模的工业标配，理解现代精排的起点 |

---

## 1. 推荐系统的基本链路

工业推荐通常分四级：

- **召回**：从百万到十亿级物品中粗筛出千级候选（双塔、图、树索引、生成式检索）。
- **粗排**：轻量模型把千级压到百级。
- **精排**：重模型预估 CTR/CVR/时长等多目标，输出百级。
- **重排**：处理多样性、新鲜度、业务规则、EE（探索利用）。

评价上，离线看 AUC / NDCG / Recall@K / HitRate，在线看 CTR、停留时长、转化、DAU 等。**推荐系统的真正难点往往不在模型结构，而在数据（行为稀疏、噪声、位置偏差）、目标（多目标冲突、长期收益）与系统（时延、算力、一致性）**。理解这一点，才能理解为什么"更简单的结构 + 更好的特征/目标/系统"经常打赢"更花哨的模型"。

---

## 2. 深度学习时代的基线算法（2016–2021）

这一代算法奠定了今天所有工业推荐的骨架，也是生成式推荐做对比实验时的 baseline。

| 算法 | 年份 / 机构 | 核心思想 | 为什么有影响力 |
| --- | --- | --- | --- |
| MF / BPR | 2009 | 隐向量内积；pairwise 排序损失 | 协同过滤的现代起点 |
| FM | 2010 | 二阶特征交叉做隐向量分解 | 稀疏特征交叉的经典解 |
| Wide&Deep | 2016，Google | 线性"记忆"分支 + 深层"泛化"分支联合训练 | 工业推荐"宽深结合"的开端 |
| YouTubeDNN | 2016，Google | 双塔召回 + sampled softmax + example age | 深度学习召回的开山之作 |
| DeepFM | 2017，华为 | FM 与 MLP 共享 embedding，端到端 | 免手工特征交叉，长期强基线 |
| DCN-V2 | 2020，Google | 显式特征交叉网络（cross network） | 把交叉结构做成可堆叠模块 |
| DIN | 2018，阿里 | 候选感知的注意力建模用户历史行为 | 用户兴趣建模的里程碑，工业精排标配 |
| DIEN | 2019，阿里 | GRU + AUGRU 建模兴趣演化 | 行为序列的时序建模 |
| SIM | 2020，阿里 | 终身行为序列，GSU 检索 + ESU 精算两阶段 | 超长序列在工业场景真正可落地 |
| MMoE | 2018，Google | 多门控专家网络 | 多目标建模的标配结构 |
| PLE | 2020，腾讯 | 分层专家 + 渐进式分离 | 缓解多任务"跷跷板"效应 |
| ESMM | 2018，阿里 | 全空间建模 CVR（CTR×CVR=CTCVR） | 解决样本选择偏差的经典工作 |
| TDM | 2018，阿里 | 树结构 + 向量检索做大规模召回 | 把召回变成可学习问题 |
| MIND | 2019，阿里 | 胶囊网络做多兴趣表示 | 多兴趣召回的代表 |
| SASRec | 2018 | 单向自注意力做 next-item 预测 | 序列推荐基石，今天仍是最常用 baseline |
| BERT4Rec | 2019 | 双向 Transformer + cloze 目标 | 双向序列建模 |
| LightGCN | 2020 | 去掉非线性与特征变换的线性图卷积 | 图推荐极简强基线 |
| PinSage | 2018，Pinterest | GraphSAGE + 随机游走，十亿级图 | 图神经网络工业召回的代表 |
| DLRM | 2019，Meta | 面向硬件特性设计的推荐架构 | 模型与系统协同设计的开端 |

**这一代的共同瓶颈**：参数主要堆在 embedding 表（可达 TB 级），而网络本体长期停留在千万参数级别，继续加宽加深收益很快饱和。这正是生成式推荐要解决的问题。

---

## 3. 生成式推荐：当前最有影响力的方向（2023–2026）

### 3.1 语义 ID（Semantic ID）：把物品变成"可生成的词"

**TIGER**（Google，NeurIPS 2023）是这一范式的起点。它用 RQ-VAE 把物品的内容 embedding 量化成由若干码字组成的「语义 ID」（如 `<12, 23, 8>`），再让 Transformer 自回归地生成下一件物品的语义 ID。

为什么重要：

- 物品 ID 从无意义的 one-hot 变成**带语义的离散 token**，模型可以把语义相似性迁移到冷启动物品上；
- 检索从"向量近邻搜索"变成"序列生成"，天然适合自回归大模型；
- 为后续所有"LLM + 推荐"的工作提供了接口。

2025 年的 *Generative Recommendation with Semantic IDs: A Practitioner's Handbook* 系统梳理了语义 ID 的构造、冲突与坍缩问题，并给出统一评测框架，说明这个方向已经从"点状创新"进入"工程化"阶段。

### 3.2 HSTU：推荐也有 scaling law

**HSTU**（Meta，ICML 2024，*Actions Speak Louder than Words*）把推荐重新表述为**序列转导（sequential transduction）**任务，并设计了适配高基数、非平稳流式数据的 HSTU 层。论文报告：

- 在公开数据集上 NDCG 最高提升 **65.8%**；
- 在 8192 长度序列上比 FlashAttention2 的 Transformer 快 **5.3–15.2 倍**；
- 模型扩到 **1.5 万亿参数**，在线 A/B 指标提升 **12.4%**，并部署到十亿级用户平台的多个场景；
- 模型质量随参数量/数据量呈幂律提升——**推荐的 scaling law**。

这是"生成式推荐"从学术概念变成工业事实的关键一步：它说明推荐系统的瓶颈不只是特征工程，而是**能不能用算力换取效果**。

### 3.3 OneRec：端到端统一召回与排序

**OneRec**（快手，2025）进一步把"召回 → 排序"的级联结构换成一个统一的生成模型：

- encoder 编码用户历史行为，decoder 逐段生成候选视频的语义 ID；
- **session-wise generation**：一次生成一个会话的推荐列表，而不是逐点生成再手工拼装；
- 用稀疏 MoE 扩大容量而不成比例增加 FLOPs；
- 用奖励模型 + DPO 做**迭代偏好对齐**（推荐场景一次曝光机会，无法同时拿到正负样本，这是与 NLP 的 DPO 最大的不同）。

论文称这是首个在真实场景显著超越复杂级联系统的端到端生成模型，在快手主场景上线后**观看时长 +1.6%**。

**OneRec-V2**（2025-08）解决了 V1 的两个工程问题：97.66% 的算力花在序列编码而非生成上，以及纯奖励模型做 RL 的局限。V2 用 lazy decoder-only 架构把总计算量降低 **94%**、训练资源降低 **90%**，从而把模型扩到 **8B 参数**，并引入基于真实交互的时长奖励塑形与自适应 ratio clipping。

### 3.4 RankMixer：排序侧的可扩展架构

**RankMixer**（字节跳动，2025）关注的是精排模型怎么"变大而不变慢"：

- 用 multi-head token mixing 替代二次复杂度的 self-attention，保留 Transformer 的并行性；
- per-token FFN 同时建模特征子空间与跨子空间交互；
- 稀疏 MoE 版本扩到 **10 亿参数**，配合动态路由解决专家训练不足与不均衡；
- 把模型 MFU 从 **4.5% 提升到 45%**，参数量扩大 **100 倍**而推理延迟基本不变；
- 1B 稠密参数版本全量上线，**用户活跃天数 +0.3%、总使用时长 +1.08%**。

它的意义在于：**生成式/大模型化的收益不只来自"生成"，也来自把排序模型改造成 GPU 友好的可扩展结构**。

### 3.5 把 LLM 接进来：三条路线

1. **LLM 当语义编码器**：用 LLM 表征构造语义 ID 或用户/物品画像。TIGER 之后的 LC-Rec、EAGER、PLUM 都在这条线上。
2. **LLM 当生成式推荐器**：**PLUM**（Google/YouTube，WWW 2026）把预训练 LLM 通过语义 ID 物品分词、领域继续预训练（CPT）、推荐任务微调三步改造成工业级生成式召回模型，在 YouTube 大规模视频数据上显著超过依赖大 embedding 表的重优化生产模型，并给出检索侧的 scaling 研究。**P5**（2022）与 **TALLRec**（2023）则是学术侧"推荐即语言任务"与"轻量指令微调"的代表。
3. **LLM 当推理/Agent**：**RecGPT-V2**（阿里，2025-12）把推荐从"隐式行为匹配"转向"显式意图推理"，用分层多智能体做意图挖掘、元提示做解释生成、约束强化学习对齐多目标，在淘宝 A/B 上 **CTR +2.98%、IPV +3.71%、TV +2.19%、NER +11.46%**，同时 GPU 消耗降低 60%、独家召回率从 9.39% 提升到 10.99%。

另外，Meta 的 **Foundation–Expert 范式**（2025）给出了"一个推荐基础模型 + 多个场景专家"的部署方案：中心 FM 产出 target-aware embedding 喂给各场景轻量专家，迁移比达到 0.64–1.0，自 2025 年起在 Meta 全量服务每天数百亿请求。这条路线回答的是"scaling law 怎么在多个业务场景分摊成本"。

### 3.6 2026 年的新动向：推理、Agent 与"生成 + 排序"混合

- **推理式推荐**：OneRec 团队后续的 OneRec-Think、OpenOneRec、OneReason 把语义 ID 与自然语言放进同一表示空间，尝试用 CoT 推理提升推荐。值得注意的是，他们在 2026 年的挑战赛总结中明确提到：**引入 CoT 并不总是提升效果**，需要加强语义对齐，否则推理会放大噪声。
- **Agentic 推荐**：2026 年出现多篇"agentic recommender systems"综述，把推荐从"打分"扩展成"规划—调用工具—多轮交互"的工作流。这更像产品形态的演进，而不是单一模型的替换。
- **生成 + 排序混合**：WWW 2026 的 GRank 等工作尝试用"generate-rank"框架兼顾生成式检索的效率与排序的精度，说明"纯端到端生成"与"级联"之间仍在互相吸收。

---

## 4. 争议与开放问题

1. **Scaling law 是否普遍成立？**
   HSTU 与 OneRec 证明了"在特定数据规模与架构下"成立；但 *Scaling New Frontiers*（2024）与 *Unlocking Scaling Law with a Three-Step Paradigm*（2025）提醒：推荐的数据是非平稳、强噪声、带反馈回路的，不能照搬 LLM 的结论。**推荐侧的 scaling 更像"数据质量 × 架构效率 × 目标对齐"的联合函数**，而不是单纯的参数量幂律。

2. **端到端生成 vs 级联架构**
   生成式在部分场景已超级联，但级联的可控性、可解释性、冷启动兜底、在线时延仍是硬约束。短期更现实的形态可能是"生成式召回 + 强排序"或"生成式重排"。

3. **语义 ID vs 原始 ID**
   语义 ID 带来泛化与冷启动收益，但量化过程会损失协同信号，也存在码本冲突、码字坍缩、长尾物品分不到好码的问题。Handbook 一类工作正在把这些问题标准化。

4. **LLM 推理是否值得？**
   RecGPT-V2 用工程手段（多智能体、压缩表示、约束 RL）把成本打下来才换来线上收益；而 CoT 在推荐上"不总有效"已被工业界公开承认。**推理的收益必须用延迟和成本来折算。**

5. **评测不可比**
   不同论文的数据集、负采样、候选集、指标口径差异极大。生成式推荐的"离线提升"经常无法复现到线上，这本身是当前最需要基础设施投入的方向之一。

---

## 5. 如果现在要做研究或落地

**研究侧：**

- 复现最小闭环：SASRec/BERT4Rec baseline → RQ-VAE 语义 ID → TIGER 式生成式检索，成本可控且能快速理解范式。
- 在小规模公开数据上验证 HSTU 的 scaling 曲线，重点看"数据量 vs 参数量"谁更关键。
- 语义 ID 的码本设计（冲突率、利用率、层级）仍是低垂果实。
- 长行为序列 + 生成式建模、生成结果的多样性与可控性、在线长期收益评估，都是开放问题。

**落地侧：**

- 不要一上来就替换级联。先用语义 ID 增强召回（尤其冷启动与跨域），再评估统一生成。
- 排序侧的结构改造（GPU 友好、MFU 提升）往往比换模型更早产生 ROI，RankMixer 是很好的参考。
- LLM 优先用在意图理解、解释生成、标签/画像生产，而不是直接做全量打分。
- 上线前务必做时延与成本核算：生成式推荐的成本结构与传统 DLRM 完全不同。

**值得跟踪的会议与来源**：RecSys、KDD、WWW、SIGIR、NeurIPS 的推荐方向（2025–2026），以及快手、Meta、Google、阿里、字节的工业技术报告。

---

## 6. 参考文献

**经典基线**

1. Rendle et al. *BPR: Bayesian Personalized Ranking from Implicit Feedback*, UAI 2009.
2. Koren et al. *Matrix Factorization Techniques for Recommender Systems*, IEEE Computer 2009.
3. Rendle. *Factorization Machines*, ICDM 2010.
4. Cheng et al. *Wide & Deep Learning for Recommender Systems*, 2016. [arXiv:1606.07792](https://arxiv.org/abs/1606.07792)
5. Covington et al. *Deep Neural Networks for YouTube Recommendations*, RecSys 2016.
6. Guo et al. *DeepFM: A Factorization-Machine based Neural Network for CTR Prediction*, IJCAI 2017. [arXiv:1703.04247](https://arxiv.org/abs/1703.04247)
7. Zhou et al. *Deep Interest Network for Click-Through Rate Prediction*, KDD 2018. [arXiv:1706.06978](https://arxiv.org/abs/1706.06978)
8. Zhou et al. *Deep Interest Evolution Network*, AAAI 2019.
9. Pi et al. *Search-based User Interest Modeling with Lifelong Sequential Behavior Data*, CIKM 2020. [arXiv:2006.05639](https://arxiv.org/abs/2006.05639)
10. Ma et al. *Modeling Task Relationships in Multi-task Learning with Multi-gate Mixture-of-Experts*, KDD 2018.
11. Tang et al. *Progressive Layered Extraction (PLE)*, RecSys 2020.
12. Ma et al. *Entire Space Multi-Task Model (ESMM)*, SIGIR 2018.
13. Zhu et al. *Learning Tree-based Deep Model for Recommender Systems*, KDD 2018. [arXiv:1801.02294](https://arxiv.org/abs/1801.02294)
14. Li et al. *Multi-Interest Network with Dynamic Routing (MIND)*, CIKM 2019.
15. Kang & McAuley. *Self-Attentive Sequential Recommendation (SASRec)*, ICDM 2018. [arXiv:1808.09781](https://arxiv.org/abs/1808.09781)
16. Sun et al. *BERT4Rec*, CIKM 2019. [arXiv:1904.06690](https://arxiv.org/abs/1904.06690)
17. He et al. *LightGCN*, SIGIR 2020. [arXiv:2002.02126](https://arxiv.org/abs/2002.02126)
18. Ying et al. *PinSage*, KDD 2018.
19. Wang et al. *DCN V2*, WWW 2021. [arXiv:2008.13535](https://arxiv.org/abs/2008.13535)
20. Naumov et al. *Deep Learning Recommendation Model (DLRM)*, 2019.

**生成式推荐与 LLM**

21. Rajput et al. *Recommender Systems with Generative Retrieval (TIGER)*, NeurIPS 2023. [arXiv:2305.05065](https://arxiv.org/abs/2305.05065)
22. Zhai et al. *Actions Speak Louder than Words: Trillion-Parameter Sequential Transducers for Generative Recommendations (HSTU)*, ICML 2024. [arXiv:2402.17152](https://arxiv.org/abs/2402.17152)
23. Deng et al. *OneRec: Unifying Retrieve and Rank with Generative Recommender and Iterative Preference Alignment*, 2025. [arXiv:2502.18965](https://arxiv.org/abs/2502.18965)
24. *OneRec-V2 Technical Report*, 2025. [arXiv:2508.20900](https://arxiv.org/abs/2508.20900)
25. Zhu et al. *RankMixer: Scaling Up Ranking Models in Industrial Recommenders*, 2025. [arXiv:2507.15551](https://arxiv.org/abs/2507.15551)
26. He et al. *PLUM: Adapting Pre-trained Language Models for Industrial-scale Generative Recommendations*, WWW 2026. [arXiv:2510.07784](https://arxiv.org/abs/2510.07784)
27. *Realizing Scaling Laws in Recommender Systems: A Foundation–Expert Paradigm*, 2025. [arXiv:2508.02929](https://arxiv.org/abs/2508.02929)
28. *RecGPT-V2 Technical Report*, 2025. [arXiv:2512.14503](https://arxiv.org/abs/2512.14503)
29. *Generative Recommendation with Semantic IDs: A Practitioner's Handbook*, 2025. [arXiv:2507.22224](https://arxiv.org/abs/2507.22224)
30. Geng et al. *Recommendation as Language Processing (P5)*, RecSys 2022. [arXiv:2203.13366](https://arxiv.org/abs/2203.13366)
31. Bao et al. *TALLRec*, 2023. [arXiv:2305.00447](https://arxiv.org/abs/2305.00447)
32. *KUAISHOU Explorer LLM-Rec Challenge 2026: Reasoning Generative Recommendation*, 2026. [arXiv:2609.39828](https://arxiv.org/abs/2609.39828)

**综述与评测**

33. Wu et al. *A Survey on Large Language Models for Recommendation*, 2023. [arXiv:2305.19860](https://arxiv.org/abs/2305.19860)
34. Guo et al. *Scaling New Frontiers: Insights into Large Recommendation Models*, 2024. [arXiv:2412.00714](https://arxiv.org/abs/2412.00714)
35. *Unlocking Scaling Law in Industrial Recommendation Systems with a Three-step Paradigm based Large User Model*, 2025. [arXiv:2502.08309](https://arxiv.org/abs/2502.08309)
36. *Think before Recommendation: Autonomous Reasoning-enhanced Recommender*, NeurIPS 2025. [arXiv:2510.23077](https://arxiv.org/abs/2510.23077)
37. *Agentic Recommender Systems: A Systematic Literature Review*, Information Processing & Management, 2026. [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0306457326005546)
38. *GRank: Towards Target-Aware and Streamlined Industrial Retrieval with a Generate-Rank Framework*, WWW 2026. [ACM DL](https://dl.acm.org/doi/10.1145/3774904.3792810)
