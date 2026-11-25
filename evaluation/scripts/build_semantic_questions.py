"""定向语义改写的成对诊断集；冻结后才运行检索，保留原题和金标。"""
from __future__ import annotations

import copy
import csv
import json
import re

from enterprise_runtime import ROOT, digest, write_json

PARENT_SHA = '1c1f063b0056c5246adbfeed04f5ad7f388f9088de1163dd4f3d59da497709eb'
PARAPHRASES = {
    'EV01': '报表按组汇总时，执行步骤里那个用散列表并行归并结果的环节特别吃工作空间。除了扩大机器资源，聚合方式上能试哪一种替代实现？',
    'EV02': '应用一次攒了很多条更新，准备最后统一提交。预留工作区时，能按准备提交的数据体积等量估算吗，还是需要额外余量？一次攒得太多应怎样改？',
    'EV03': '业务程序取一份很大的结果时自己先撑爆了，数据库机器却没异常。应该排查连接驱动的什么取数设置，防止把整份数据囤在本地？',
    'EV04': '数据库内置那组专门展示事务互相阻塞关系的观测表，能覆盖所有并发控制方式吗？里面展示的请求文本，还能还原成带着当时真实入参和排版的原样吗？',
    'EV05': '同一条业务记录被大量请求争着修改，先占住它再更新的模式下，反复尝试占用的次数已经耗尽。有没有让争抢更公平的开关，开启后处理速度可能付出什么代价？',
    'EV06': '一个先占住待改行、到结束才释放的更新任务运行很久，后台维持这些占用有效性的管理器提示已超时、保护可能失效。应核对哪个最长存活设置，业务流程可以怎样改？',
    'EV07': '一条读写请求总共用了三十毫秒，其中二十五毫秒都在等领到本次操作的全局版本号。这个占比是否值得警惕，应该优先检查哪个准备阶段？',
    'EV08': '接口等了将近八秒，执行明细却只记了几毫秒。日志里开跑前改写请求的准备环节很慢，嵌在请求里的独立取数也在那时花掉了大部分时间。这种耗时没进执行明细的情况怎样解释？',
    'EV09': '任务分发到好几台存储机器后，大多数只排队一两毫秒，有一台却等了大约十分之一秒。记录里给了等待最久的那台地址，应该优先朝什么方向排查，而不是只看平均数？',
    'EV10': '数据库接入层测到从发出访问到收到回复隔了很久，存储进程自己测到的处理过程却很快。两边计时差这么多，除了计算能力，还应该怀疑哪一段？',
    'EV11': '组件起不来，系统提示能同时持有的文件句柄名额用完了。启动前该查看哪一项操作系统额度？',
    'EV12': '第二个存储副本进程准备启动时，沿用了第一个实例正在使用的持久化路径，提示文件已被占住。能让两套进程共用那一套数据库文件吗？',
    'EV13': '直接运行可执行文件部署时，保存数据和负责调度的两个组件改了磁盘上的参数清单，重启却没按新值工作。应检查有没有把清单位置传给程序？调度组件以后每次重启都会重读吗？',
    'EV14': '用拷贝工具测盘时，担心漂亮的速度其实只是数据进了操作系统缓冲区。要求写入直接送到设备的那个选项，为什么更适合测物理盘的本事？',
    'EV15': '同事说采集还在正常上报，我用自己的电脑打开仪表盘，却缺了部分曲线。除了采集链路，有没有与两边钟表没对齐有关的检查项？',
    'EV16': '应用换到分布式数据库后，一次没指定先后规则的取数，行的出场次序和以前不一样。能靠原来单机上碰巧稳定的次序继续用吗？怎样把先后变成明确约定？',
    'EV17': '批量清掉旧记录后，盘占用没跟着立刻缩小。回收器处理过期版本后，底层还得做哪一步整理，才能把这部分存储重新利用起来？',
    'EV18': '线上大表想给现有记录补建一套用于快速查找的目录，团队怕这段时间读写全部停住。这个后台过程为什么会慢，会不会额外抢资源，是否等于必须停服？',
    'EV19': '只把存放集群元信息的调度服务重建了，保存业务记录的进程还拿着旧目录重启。为什么它会因新旧集群身份对不上而拒绝加入？',
    'EV20': '公司要为两千 GB 的库留一份备份，现有资料优先推荐哪个原生工具？拿通用的 MySQL 逻辑导出程序搬这么多数据，有什么效率问题？',
    'EV21': '原先跑在 MySQL 上的业务要搬进 TiDB，希望先迁数据，再一批批把读和写切过去，前面放个固定入口少改应用。搬数据的工具和接入层应该怎样配合？',
    'EV22': '观察各台机器忙着读写盘的程度时，一种情况是少数主机显著高于同伴，另一种是几乎所有主机都很忙。这两种分布下，排查范围应有什么区别？',
    'EV23': '有人建议为新增记录采用服务端分配、随机打散的唯一编号，减轻都写向同一处的压力。业务还靠编号大小判断谁先创建，这能继续成立吗？写入后用什么方式取回刚分配的编号？',
    'EV24': '我已经给会话设好了默认的事务控制方式，但开启某次操作时又明确声明用另一种锁策略。数据库最终听默认参数，还是听这次开头那句显式声明？',
}


def main():
    parent_path=ROOT/'evaluation/datasets/enterprise_tidb_v2_frozen/questions.json'
    parent_bytes=parent_path.read_bytes()
    if digest(parent_path)!=PARENT_SHA:raise ValueError('Parent frozen benchmark changed')
    parent=json.loads(parent_bytes)
    original={q['id']:q for q in parent['questions'] if q['split']=='test'}
    if set(original)!=set(PARAPHRASES):raise ValueError('Paired question coverage differs')
    dataset=copy.deepcopy(parent)
    dataset.update(dataset_id='enterprise_semantic_paraphrase_v1',purpose='定向语义改写与措辞鲁棒性诊断，不是随机生产问题分布或新的独立业务留出',
                   provenance='AI 在已知原题和结果后设计的成对改写；新问法在查看其检索排名前统一固定，全部24对保留',
                   parent_frozen_dataset=parent_path.relative_to(ROOT).as_posix(),parent_frozen_dataset_sha256=PARENT_SHA,
                   counts={'test':24},question_count=24,
                   selection_policy='同一24题只改问法，不替换语料、答案、金标、检索参数或删除失败题；回避可替换的报错原文和参数拼写，保留问题成立的条件')
    dataset['questions']=[]
    for qid,old in original.items():
        q=copy.deepcopy(old)
        q.update(original_question=old['question'],question=PARAPHRASES[qid],variant_id='SP'+qid[2:],
                 query_style='口语化业务现象／术语释义；部分必要产品名保留')
        assert q['reference_answer']==old['reference_answer'] and q['evidence_mapping']==old['evidence_mapping'] and q['gold_chunk_ids']==old['gold_chunk_ids']
        if not q['question'] or q['question']==old['question']:raise ValueError('Invalid paraphrase')
        dataset['questions'].append(q)
    destination=ROOT/'evaluation/datasets/enterprise_semantic_paraphrase_v1'
    destination.mkdir(parents=True,exist_ok=True)
    path=destination/'questions.json'
    data=(json.dumps(dataset,ensure_ascii=False,indent=2)+'\n').encode('utf-8')
    if path.exists() and path.read_bytes()!=data:raise ValueError('Frozen semantic version exists; do not overwrite')
    path.write_bytes(data)
    path.with_suffix('.sha256').write_text(digest(path)+'\n',encoding='ascii')
    with (destination/'paired_questions.csv').open('w',encoding='utf-8-sig',newline='') as handle:
        writer=csv.writer(handle);writer.writerow(['id','original_question','semantic_question','reference_answer','kind'])
        for q in dataset['questions']:writer.writerow([q['variant_id'],q['original_question'],q['question'],q['reference_answer'],q['query_style']])
    report={'question_pairs':24,'parent_sha256':PARENT_SHA,'semantic_sha256':digest(path),
            'answers_and_gold_ids_unchanged':True,'all_pairs_retained':True,'reindexing_required':False,
            'new_question_rankings_seen_during_authoring':False,'ranking_based_filtering':False,
            'interpretation':'Targeted paraphrase robustness set, adapted from already exposed tasks; not independent holdout or corpus optimization',
            'paid_model_calls':0}
    write_json(destination/'freeze_report.json',report)
    assert parent_path.read_bytes()==parent_bytes
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
