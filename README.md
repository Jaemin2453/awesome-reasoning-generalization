<div align="center">

# Awesome Reasoning Generalization

### Beyond the Training Distribution: A Survey of Reasoning Generalization in Large Language Models

[![Awesome](https://awesome.re/badge.svg)](https://awesome.re)
![Papers](https://img.shields.io/badge/papers-239-6f42c1)
![2024 papers](https://img.shields.io/badge/2024%20papers-35-yellow)
![2025 papers](https://img.shields.io/badge/2025%20papers-71-orange)
![2026 papers](https://img.shields.io/badge/2026%20papers-112-red)
[![GitHub last commit](https://img.shields.io/github/last-commit/tue09/awesome-reasoning-generalization?logo=github&color=blue)](https://github.com/tue09/awesome-reasoning-generalization/commits/main)

</div>

> **Status:** Updated on 13 October 2026 to follow the complete version of the survey (literature search through 22 September 2026). The corpus version contains 239 studies, of which 173 are cited in the current survey: 72 on training, 24 on inference, 22 on architecture, and 55 on analysis. These counts include contextual methods as well as direct transfer evidence. The remaining 66 studies are retained in the catalogue. By first arXiv posting, 112 are from 2026, 71 from 2025, 35 from 2024, and 21 are earlier foundations.

Progress in LLM reasoning is commonly measured on benchmarks that stay close to the training distribution. This list collects work on the question the survey asks: when and how can large language models generalize their reasoning beyond their training distributions, under changes in notation, problem structure, length, difficulty, domain, language, or tool interface? The companion text is in [survey.md](survey.md), and the paper is in [full_paper/](full_paper/).

## Contents

- [Reading a Generalization Claim](#reading-a-generalization-claim)
- [Taxonomy](#taxonomy)
- [Paper List](#paper-list)
  - [Training for Generalization](#training-for-generalization)
    - [Pre-Training](#pre-training): [Data Curation](#data-curation), [Optimizer Design](#optimizer-design)
    - [Mid-Training](#mid-training): [Continued Pre-Training](#continued-pre-training), [Intermediate Fine-Tuning](#intermediate-fine-tuning)
    - [Post-Training: Supervised Fine-Tuning](#post-training-supervised-fine-tuning): [Problem-Level Methods](#problem-level-methods), [Solution-Level Methods](#solution-level-methods), [Objective-Level Methods](#objective-level-methods)
    - [Post-Training: Reinforcement Learning](#post-training-reinforcement-learning): [Environment Design](#environment-design), [Reward Design](#reward-design), [Policy Optimization](#policy-optimization)
    - [Post-Training: Hybrid Training](#post-training-hybrid-training): [Sequential Training](#sequential-training), [Joint Training](#joint-training)
  - [Inference for Generalization](#inference-for-generalization)
    - [Trajectory Restructuring](#trajectory-restructuring): [Structural Decomposition](#structural-decomposition), [Upfront Structural Injection](#upfront-structural-injection)
    - [Compute Scaling](#compute-scaling): [Search Expansion and Allocation](#search-expansion-and-allocation), [Targeted Revision](#targeted-revision)
    - [State Assessment](#state-assessment): [Internal State Assessment](#internal-state-assessment), [Verifier-Mediated Assessment](#verifier-mediated-assessment)
    - [Inference-Time Externalization](#inference-time-externalization): [Within-Episode Externalization](#within-episode-externalization), [Cross-Episode Externalization](#cross-episode-externalization)
  - [Architecture for Generalization](#architecture-for-generalization)
    - [Recurrent Depth](#recurrent-depth): [Weight Sharing](#weight-sharing), [Adaptive Computation](#adaptive-computation)
    - [Information Routing](#information-routing): [Positional Encoding](#positional-encoding), [Attention Patterns](#attention-patterns), [Modular Reasoning](#modular-reasoning)
    - [Recurrent Memory](#recurrent-memory): [Token-Level Recurrence](#token-level-recurrence), [Chunk-Level Recurrence](#chunk-level-recurrence)
  - [Analysis of Generalization](#analysis-of-generalization)
    - [Behavioral Evidence](#behavioral-evidence): [Compositional Generalization](#compositional-generalization), [Length, Depth, and Difficulty](#length-depth-and-difficulty), [Prior Exposure](#prior-exposure)
    - [Mechanistic Evidence](#mechanistic-evidence): [Shared Circuits](#shared-circuits), [Interfaces Between Steps](#interfaces-between-steps), [Learning Dynamics](#learning-dynamics)
    - [Theoretical Evidence](#theoretical-evidence): [Learnability](#learnability), [Certification](#certification)
- [Datasets and Benchmarks](#datasets-and-benchmarks)
- [Related Surveys](#related-surveys)
- [Citation](#citation)

## Reading a Generalization Claim

A reasoning-generalization result can be interpreted only when it states what the model has encountered, what changes at test time, and which components remain fixed. Table 5 of the survey lists the minimum report.

| Field | Minimum report | Misreading prevented |
| --- | --- | --- |
| **Reasoning task** | Generator or dataset, target output, metric, and relevant constraints | Treating a broad benchmark score as evidence about one reasoning process |
| **Reference exposure** | Pre-training knowledge if available; all continued-training, SFT, distillation, RL, and adaptation data | Calling an item unseen only because it was absent from the last training stage |
| **Evaluated components** | Whether weights, prompts, exemplars, controller, verifier, memory, retriever, and tools remain fixed | Attributing gains from inference methods or external resources to changes in learned model capabilities |
| **Test shift** | Named axis, construction method, distance or level, and matched ID control | Combining instance, surface, composition, length, and domain shifts into one OOD label |
| **Outcome and evidence** | Performance with uncertainty, generalization gap when comparable, mechanism test if claimed | Treating a probe, score change, or finite test range as stronger evidence than it provides |
| **Inference cost** | Samples, generated tokens, tool calls, wall time or FLOPs, and stopping rule | Reporting an accuracy gain that follows only from unbounded extra computation |

Gains enabled by prompting, search, verification, or memory need not hold for the model alone: a tool-augmented system can solve longer problems even when the base model has not learned a length-general algorithm. Greedy accuracy shows what a model produces reliably, large-pass@k accuracy what it can produce occasionally, and verifier-selected accuracy what a larger system can recover.

## Taxonomy

| Pillar | Central question | Branches |
| --- | --- | --- |
| **Training for Generalization** | How can parameter updates make a learned reasoning procedure survive shifts in form, composition, length, difficulty, domain, or language? | Pre-training, mid-training, and post-training (SFT, RL, hybrid) |
| **Inference for Generalization** | How can test-time computation, verification, and adaptation support transfer to unfamiliar reasoning problems? | Trajectory restructuring, compute scaling, state assessment, externalization |
| **Architecture for Generalization** | Which computational structures let a learned operation be reused when problems grow longer or combine familiar steps in new ways? | Recurrent depth, information routing, recurrent memory |
| **Analysis of Generalization** | What actually transfers under distribution shift, and why? | Behavioral, mechanistic, and theoretical evidence |

<p align="center">
  <a href="assets/main_taxonomy.pdf"><img src="assets/main_taxonomy.svg" width="100%" alt="Taxonomy of reasoning generalization in large language models"></a>
</p>

<p align="center"><em>Figure 1 of the survey: taxonomy of reasoning generalization in LLMs, with representative works for each branch.</em></p>

The first three pillars are interventions, ordered by where they act: on parameters, on test-time computation with the parameters fixed, and on the model's computational structure. Training is organized by stage and, within each stage, by the component a method modifies. The fourth pillar separates behavioral, mechanistic, and theoretical evidence. Each paper appears once, under its primary source of generalization; studies the survey assigns to a branch as a whole, such as evidence on RL transfer, are listed before that branch's leaves.

## Paper List

### Training for Generalization

*How can parameter updates make a learned reasoning procedure survive shifts in form, composition, length, difficulty, domain, or language?*

#### Pre-Training

*Pre-training provides the model with its initial knowledge and reasoning primitives.*

##### Data Curation

> Control what the corpus contains: the primitives that later compositions require and the share of facts that must be inferred rather than stated.

- **On the Interplay of Pre-Training, Mid-Training, and RL on Reasoning Language Models**. *Charlie Zhang, Graham Neubig, Xiang Yue*. [[Paper]](https://arxiv.org/abs/2512.07783) [[PDF]](https://arxiv.org/pdf/2512.07783) ![](https://img.shields.io/badge/year-2025-orange)
- **Grokking in the Wild: Data Augmentation for Real-World Multi-Hop Reasoning with Transformers**. *Roman Abramov, Felix Steinbauer, Gjergji Kasneci*. [[Paper]](https://arxiv.org/abs/2504.20752) [[PDF]](https://arxiv.org/pdf/2504.20752) ![](https://img.shields.io/badge/year-2025-orange)
- **Teaching Transformers Causal Reasoning through Axiomatic Training**. *Aniket Vashishtha, Abhinav Kumar, Atharva Pandey, Abbavaram Gowtham Reddy, Kabir Ahuja, Vineeth N Balasubramanian, Amit Sharma*. [[Paper]](https://arxiv.org/abs/2407.07612) [[PDF]](https://arxiv.org/pdf/2407.07612) ![](https://img.shields.io/badge/year-2024-yellow)

##### Optimizer Design

> Change how the model fits the same corpus, by aligning gradients across data sources or by favoring primitives over memorization.

- **Nexus: Same Pretraining Loss, Better Downstream Generalization via Common Minima**. *Huanran Chen, Huaqing Zhang, Xiao Li, Yinpeng Dong, Ke Shen, Jun Zhu*. [[Paper]](https://arxiv.org/abs/2604.09258) [[PDF]](https://arxiv.org/pdf/2604.09258) ![](https://img.shields.io/badge/year-2026-red)
- **Complexity Control Facilitates Reasoning-Based Compositional Generalization in Transformers**. *Zhongwang Zhang, Pengxiao Lin, Zhiwei Wang, Yaoyu Zhang, Zhi-Qin John Xu*. [[Paper]](https://arxiv.org/abs/2501.08537) [[PDF]](https://arxiv.org/pdf/2501.08537) ![](https://img.shields.io/badge/year-2025-orange)
- **Initialization is Critical to Whether Transformers Fit Composite Functions by Reasoning or Memorizing**. *Zhongwang Zhang, Pengxiao Lin, Zhiwei Wang, Yaoyu Zhang, Zhi-Qin John Xu*. [[Paper]](https://arxiv.org/abs/2405.05409) [[PDF]](https://arxiv.org/pdf/2405.05409) ![](https://img.shields.io/badge/year-2024-yellow)

#### Mid-Training

*A stage between pre-training and task-specific adaptation that narrows the gap between their distributions and supplies skills the target tasks require.*

##### Continued Pre-Training

> Next-token prediction on raw text from a domain, or a positional range, under-represented in pre-training.

- **Randomized YaRN Improves Length Generalization for Long-Context Reasoning**. *Manas Mehta, Fangcong Yin, Greg Durrett*. [[Paper]](https://arxiv.org/abs/2606.23687) [[PDF]](https://arxiv.org/pdf/2606.23687) ![](https://img.shields.io/badge/year-2026-red)
- **How Post-Training Shapes Biological Reasoning Models**. *Lukas Fesser, Hanlin Zhang, Michelle M. Li, Eric Wang, Bryan Perozzi, Shekoofeh Azizi, Sham M. Kakade, Marinka Zitnik*. [[Paper]](https://arxiv.org/abs/2606.16517) [[PDF]](https://arxiv.org/pdf/2606.16517) ![](https://img.shields.io/badge/year-2026-red)
- **DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models**. *Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, Y. K. Li, Y. Wu, Daya Guo*. [[Paper]](https://arxiv.org/abs/2402.03300) [[PDF]](https://arxiv.org/pdf/2402.03300) ![](https://img.shields.io/badge/year-2024-yellow)

##### Intermediate Fine-Tuning

> Training on solved problems that need no domain knowledge, so that the acquired reasoning transfers and prepares later training.

- **Mid-Training with Self-Generated Data Improves Reinforcement Learning in Language Models**. *Aswin RRV, Jacob Dineen, Divij Handa, Mihir Parmar, Ben Zhou, Swaroop Mishra, Chitta Baral*. [[Paper]](https://arxiv.org/abs/2605.08472) [[PDF]](https://arxiv.org/pdf/2605.08472) ![](https://img.shields.io/badge/year-2026-red)
- **Fundamental Reasoning Paradigms Induce Out-of-Domain Generalization in Language Models**. *Mingzi Cao, Xingwei Tan, Mahmud Elahi Akhter, Marco Valentino, Maria Liakata, Xi Wang, Nikolaos Aletras*. [[Paper]](https://arxiv.org/abs/2602.08658) [[PDF]](https://arxiv.org/pdf/2602.08658) ![](https://img.shields.io/badge/year-2026-red)
- **Atomic Skills are the Prerequisite: When Reinforcement Learning Synthesizes Compositional Reasoning, and When It Only Amplifies**. *Sitao Cheng, Xunjian Yin, Ruiwen Zhou, Yuxuan Li, Xinyi Wang, Liangming Pan, William Yang Wang, Victor Zhong*. [[Paper]](https://arxiv.org/abs/2512.01970) [[PDF]](https://arxiv.org/pdf/2512.01970) ![](https://img.shields.io/badge/year-2025-orange)
- **Warm Up Before You Train: Unlocking General Reasoning in Resource-Constrained Settings**. *Safal Shrestha, Minwu Kim, Aadim Nepal, Anubhav Shrestha, Keith Ross*. [[Paper]](https://arxiv.org/abs/2505.13718) [[PDF]](https://arxiv.org/pdf/2505.13718) ![](https://img.shields.io/badge/year-2025-orange)
- **Enhancing Reasoning Capabilities of LLMs via Principled Synthetic Logic Corpus**. *Terufumi Morishita, Gaku Morio, Atsuki Yamaguchi, Yasuhiro Sogawa*. [[Paper]](https://arxiv.org/abs/2411.12498) [[PDF]](https://arxiv.org/pdf/2411.12498) ![](https://img.shields.io/badge/year-2024-yellow)

#### Post-Training: Supervised Fine-Tuning

*Imitation of demonstrations, which tends to absorb spurious surface correlations unless the problems, solutions, or objective are changed.*

##### Problem-Level Methods

> Represent the anticipated test shift among the training problems, leaving no single surface pattern to memorize.

**Problem rewriting**

- **Mitigating Spurious Correlations in LLMs via Causality-Aware Post-Training**. *Shurui Gui, Shuiwang Ji*. [[Paper]](https://arxiv.org/abs/2506.09433) [[PDF]](https://arxiv.org/pdf/2506.09433) ![](https://img.shields.io/badge/year-2025-orange)
- **Your Language Model May Think Too Rigidly: Achieving Reasoning Consistency with Symmetry-Enhanced Training**. *Yihang Yao, Zhepeng Cen, Miao Li, William Han, Yuyou Zhang, Emerson Liu, Zuxin Liu, Chuang Gan, Ding Zhao*. [[Paper]](https://arxiv.org/abs/2502.17800) [[PDF]](https://arxiv.org/pdf/2502.17800) ![](https://img.shields.io/badge/year-2025-orange)

**Problem generation**

- **Selective Left-Shift: Turning Test-Time Compute and Difficulty-based Curation into Training Data for Low-Resource Code Generation**. *Didula Samaraweera, Anjana Supun, Srinath Perera*. [[Paper]](https://arxiv.org/abs/2607.07748) [[PDF]](https://arxiv.org/pdf/2607.07748) ![](https://img.shields.io/badge/year-2026-red)
- **Towards Compositional Generalization of LLMs via Skill Taxonomy Guided Data Synthesis**. *Yifan Wei, Li Du, Xiaoyan Yu, Yang Feng, Angsheng Li*. [[Paper]](https://arxiv.org/abs/2601.03676) [[PDF]](https://arxiv.org/pdf/2601.03676) ![](https://img.shields.io/badge/year-2026-red)
- **Self-Improving Transformers Overcome Easy-to-Hard and Length Generalization Challenges**. *Nayoung Lee, Ziyang Cai, Avi Schwarzschild, Kangwook Lee, Dimitris Papailiopoulos*. [[Paper]](https://arxiv.org/abs/2502.01612) [[PDF]](https://arxiv.org/pdf/2502.01612) ![](https://img.shields.io/badge/year-2025-orange)

**Difficulty curriculum**

- **Ladders of Thought: A Self-Evolving Curriculum of Progressively Simplified Reasoning Traces**. *Minghui Liu, Thomas Magelinski, Dehao Yuan, Qi Yu, Furong Huang*. [[Paper]](https://arxiv.org/abs/2609.25643) [[PDF]](https://arxiv.org/pdf/2609.25643) ![](https://img.shields.io/badge/year-2026-red)
- **Data Difficulty and the Generalization--Extrapolation Tradeoff in LLM Fine-Tuning**. *Siyuan Liu, Tinghong Chen, Xinghan Li, Yifei Wang, Jingzhao Zhang*. [[Paper]](https://arxiv.org/abs/2605.12906) [[PDF]](https://arxiv.org/pdf/2605.12906) ![](https://img.shields.io/badge/year-2026-red)
- **Rethinking Easy-to-Hard: Limits of Curriculum Learning in Post-Training for Deductive Reasoning**. *Maximilian Mordig, Andreas Opedal, Weiyang Liu, Bernhard Schölkopf*. [[Paper]](https://arxiv.org/abs/2603.27226) [[PDF]](https://arxiv.org/pdf/2603.27226) ![](https://img.shields.io/badge/year-2026-red)

##### Solution-Level Methods

> Rewrite the imitated solutions, since imitation generalizes only as far as its target exposes a reusable procedure.

**Abstract solutions**

- **Zipping the Thought: When and How Compressed Reasoning Data Works in LLM Post-Training**. *Kohsei Matsutani, Gouki Minegishi, Takeshi Kojima, Yusuke Iwasawa, Yutaka Matsuo*. [[Paper]](https://arxiv.org/abs/2605.28008) [[PDF]](https://arxiv.org/pdf/2605.28008) ![](https://img.shields.io/badge/year-2026-red)
- **Learning to Adapt SFT Data for Better Reasoning Generalization**. *Lisong Sun, Li Wang, Chen Zhang, Jinyang Wu, Kui Zhang, Tianhao Peng, Wenjun Wu*. [[Paper]](https://arxiv.org/abs/2605.26924) [[PDF]](https://arxiv.org/pdf/2605.26924) ![](https://img.shields.io/badge/year-2026-red)
- **Memorize Theorems, Not Instances: Probing SFT Generalization through Mathematical Reasoning**. *Ruiying Peng, Mengyu Yang, Jing Lei, Xiaohui Li, Xueyu Wu, Xinlei Chen*. [[Paper]](https://arxiv.org/abs/2605.09270) [[PDF]](https://arxiv.org/pdf/2605.09270) ![](https://img.shields.io/badge/year-2026-red)
- **From Meta-Thought to Execution: Cognitively Aligned Post-Training for Generalizable and Reliable LLM Reasoning**. *Shaojie Wang, Liang Zhang*. [[Paper]](https://arxiv.org/abs/2601.21909) [[PDF]](https://arxiv.org/pdf/2601.21909) ![](https://img.shields.io/badge/year-2026-red)

**Stepwise solutions**

- **The Imitation Game: Turing Machine Imitator is Length Generalizable Reasoner**. *Zhouqi Hua, Wenwei Zhang, Chengqi Lyu, Yuzhe Gu, Songyang Gao, Kuikun Liu, Dahua Lin, Kai Chen*. [[Paper]](https://arxiv.org/abs/2507.13332) [[PDF]](https://arxiv.org/pdf/2507.13332) ![](https://img.shields.io/badge/year-2025-orange)
- **Learning Composable Chains-of-Thought**. *Fangcong Yin, Zeyu Leo Liu, Liu Leqi, Xi Ye, Greg Durrett*. [[Paper]](https://arxiv.org/abs/2505.22635) [[PDF]](https://arxiv.org/pdf/2505.22635) ![](https://img.shields.io/badge/year-2025-orange)
- **Beyond In-Distribution Success: Scaling Curves of CoT Granularity for Language Model Generalization**. *Ru Wang, Wei Huang, Selena Song, Haoyu Zhang, Qian Niu, Yusuke Iwasawa, Yutaka Matsuo, Jiaxian Guo*. [[Paper]](https://arxiv.org/abs/2502.18273) [[PDF]](https://arxiv.org/pdf/2502.18273) ![](https://img.shields.io/badge/year-2025-orange)
- **Beyond Single-Task: Robust Multi-Task Length Generalization for LLMs**. *Yi Hu, Shijia Kang, Haotong Yang, Haotian Xu, Muhan Zhang*. [[Paper]](https://arxiv.org/abs/2502.11525) [[PDF]](https://arxiv.org/pdf/2502.11525) ![](https://img.shields.io/badge/year-2025-orange)
- **Case-Based or Rule-Based: How Do Transformers Do the Math?** *Yi Hu, Xiaojuan Tang, Haotong Yang, Muhan Zhang*. [[Paper]](https://arxiv.org/abs/2402.17709) [[PDF]](https://arxiv.org/pdf/2402.17709) ![](https://img.shields.io/badge/year-2024-yellow)

##### Objective-Level Methods

> Change the loss so that what would not hold under shift contributes less to the update.

**Teacher-based objectives**

- **Every Coin Has Two Sides: On the Dual Nature of Generalization in On-Policy Distillation of Large Language Models**. *Zhaoyi Li, Deyang Kong, Yuan Wei, Evan Yang, Ranran Shen, Mahardika Krisna Ihsani, Ming Yang, Wei Zhang, Chuan Hao, Jian Yang, et al.* [[Pap