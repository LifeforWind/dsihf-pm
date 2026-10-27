# Third-party sources and notices

This repository does not claim authorship of the upstream benchmark or baseline methods. It does not apply a blanket license to separately obtained third-party code.

| Component | Source | Revision / notice |
| --- | --- | --- |
| A-mem engineering implementation | [agiresearch/A-mem](https://github.com/agiresearch/A-mem) | Native base `ceffb860f0712bbae97b184d440df62bc910ca8d`; MIT notice preserved verbatim at `a-mem/LICENSE`. Local extensions are identified in the source manifest. |
| PM-Bench | [genglinliu/PMBench](https://github.com/genglinliu/PMBench) | `e1093c470c8981daf522d4ef047a7c3a71e077d7`; downloaded from the original source, not vendored here. No standalone code license was found in that local revision; the paper license is not asserted to license the code. |
| PIS / ProMem | [ChengcanWu/prospective-memory-reaearch](https://github.com/ChengcanWu/prospective-memory-reaearch) | `43e9b426a478bdd6b4e6fb12cb609ca26bd71ff9`; keep its own LICENSE. Its bundled decision parser is checksum-verified when extracted. |
| PMA | [yifannnwu/proactive-memory-agent](https://github.com/yifannnwu/proactive-memory-agent) | `89e5c0d6aadfe531a1aee42fd290d48be89973dd`; keep upstream notices. |
| Mem0 | [mem0ai/mem0](https://github.com/mem0ai/mem0) | `0724291bd864cf807e88ee8981c8226644e8fb42`, package 2.2.1; keep upstream LICENSE and OSS notices. |

`configs/upstreams.json` is the machine-readable dependency lock. Bootstrap stores dependencies in ignored directories and never commits their datasets, archives, original trajectories or scores to this repository. Consult the license accompanying each source for reuse. This first code organization does not choose a new blanket license for the project's original contributions.
